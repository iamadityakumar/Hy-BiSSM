import subprocess, sys
try:
    import mamba_ssm
    print('mamba_ssm already installed')
except ImportError:
    print('Installing mamba-ssm...')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                           'mamba-ssm', '--no-build-isolation', '--quiet'])

import os, random, time, warnings
from pathlib import Path
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

# Primary device — DataParallel handles both GPUs automatically
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_GPUS = torch.cuda.device_count()
print(f'Device : {DEVICE}')
print(f'GPUs   : {N_GPUS}')
for i in range(N_GPUS):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')

class CFG:
    # ── Data ──────────────────────────────────────────────────────────────────
    DATA_ROOT    = 'OHID-1'
    ALL_SCENES   = list(range(1, 11))         # scenes 1–10
    N_CLASSES    = 7
    N_BANDS      = 32
    CLASS_NAMES  = ['Background', 'Farmland', 'Building',
                    'Road', 'Water', 'Vegetation', 'Other']
    NIR_START    = 19                         # 0-indexed, inclusive

    # ── Patch sizes ───────────────────────────────────────────────────────────
    LOCAL_SIZE   = 7
    GLOBAL_SIZE  = 25
    LOCAL_PAD    = LOCAL_SIZE  // 2
    GLOBAL_PAD   = GLOBAL_SIZE // 2

    # ── Training ──────────────────────────────────────────────────────────────
    TRAIN_RATIO  = 0.10
    # 256 fills both T4s (16 GB each). Reduce to 128 if OOM.
    BATCH_SIZE   = 256
    EPOCHS       = 150
    LR           = 3e-4
    WEIGHT_DECAY = 1e-4
    PATIENCE     = 20

    # ── Model dims ────────────────────────────────────────────────────────────
    SPEC_DIM     = 64
    SPAT_DIM     = 128
    FUSED_DIM    = SPEC_DIM + SPAT_DIM       # 192

    LAMBDA_CONTRAST = 0.0
    LAMBDA_UNMIX    = 0.0
    LAMBDA_MORPH    = 0.0

    # ── Output ────────────────────────────────────────────────────────────────
    SAVE_PATH    = 'hybissm_net_all10.pt'
    PROTO_PATH   = 'all10_prototypes.pt'

# ── Data Loading & Split Helper Functions ─────────────────────────────────────

def load_ohid1_scene(cfg, scene_idx):
    """
    Loads one OHID-1 scene by scene_idx (1-10).
    Returns:
        image  : (H, W, B) float32, per-band normalised to [0,1]
        labels : (H, W)    int64,   0 = unlabelled
    """
    img_dir = Path(cfg.DATA_ROOT) / 'images'
    lbl_dir = Path(cfg.DATA_ROOT) / 'labels'

    mat_path = img_dir / f'201912_{scene_idx}.mat'
    tif_path = img_dir / f'201912_{scene_idx}.tif'

    if mat_path.exists():
        data  = sio.loadmat(str(mat_path))
        key   = [k for k in data if not k.startswith('_')][0]
        image = data[key].astype(np.float32)
        if image.shape[0] == cfg.N_BANDS:
            image = image.transpose(1, 2, 0)
    elif tif_path.exists():
        try:
            import rasterio
            with rasterio.open(str(tif_path)) as src:
                image = src.read().astype(np.float32).transpose(1, 2, 0)
        except ImportError:
            from PIL import Image as PILImage
            image = np.array(PILImage.open(str(tif_path))).astype(np.float32)
    else:
        raise FileNotFoundError(f'No image for scene {scene_idx} in {img_dir}')

    for b in range(image.shape[2]):
        mn, mx = image[:, :, b].min(), image[:, :, b].max()
        if mx > mn:
            image[:, :, b] = (image[:, :, b] - mn) / (mx - mn)

    lbl_mat = lbl_dir / f'201912_{scene_idx}.mat'
    lbl_tif = lbl_dir / f'201912_{scene_idx}.tif'

    if lbl_mat.exists():
        data   = sio.loadmat(str(lbl_mat))
        key    = [k for k in data if not k.startswith('_')][0]
        labels = data[key].astype(np.int64).squeeze()
    elif lbl_tif.exists():
        try:
            import rasterio
            with rasterio.open(str(lbl_tif)) as src:
                labels = src.read(1).astype(np.int64)
        except ImportError:
            from PIL import Image as PILImage
            labels = np.array(PILImage.open(str(lbl_tif))).astype(np.int64)
    else:
        raise FileNotFoundError(f'No labels for scene {scene_idx}')

    n_labeled = (labels > 0).sum()
    print(f'  Scene {scene_idx:>2}: image {image.shape}  labeled {n_labeled:,}')
    return image, labels

class DualPatchDataset(Dataset):
    """
    Returns (local_patch, global_patch, label, scene_id) per labeled pixel.
    Padding: reflect, computed once at init.
    """
    def __init__(self, image, labels, cfg, indices, scene_id, augment=False):
        self.cfg      = cfg
        self.labels   = labels
        self.indices  = indices
        self.scene_id = scene_id
        self.augment  = augment

        pad = cfg.GLOBAL_PAD
        self.padded = np.pad(image,
                             ((pad, pad), (pad, pad), (0, 0)),
                             mode='reflect')
        self.offset = pad

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        r, c = self.indices[idx]
        ro   = r + self.offset
        co   = c + self.offset
        lp   = self.cfg.LOCAL_PAD
        gp   = self.cfg.GLOBAL_PAD

        local_p  = self.padded[ro-lp:ro+lp+1, co-lp:co+lp+1, :]
        global_p = self.padded[ro-gp:ro+gp+1, co-gp:co+gp+1, :]

        local_p  = torch.from_numpy(local_p.transpose(2, 0, 1).copy()).float()
        global_p = torch.from_numpy(global_p.transpose(2, 0, 1).copy()).float()

        if self.augment:
            if random.random() > 0.5:
                local_p  = torch.flip(local_p,  dims=[2])
                global_p = torch.flip(global_p, dims=[2])
            if random.random() > 0.5:
                local_p  = torch.flip(local_p,  dims=[1])
                global_p = torch.flip(global_p, dims=[1])

        label = int(self.labels[r, c]) - 1   # 0-indexed
        return local_p, global_p, label, self.scene_id


def make_splits(labels, cfg):
    """Returns (train_idx, val_idx, test_idx) as lists of (row, col)."""
    rows, cols = np.where(labels > 0)
    all_idx    = list(zip(rows.tolist(), cols.tolist()))
    all_lbl    = [labels[r, c] for r, c in all_idx]

    train_idx, temp_idx, _, temp_lbl = train_test_split(
        all_idx, all_lbl,
        train_size=cfg.TRAIN_RATIO,
        stratify=all_lbl, random_state=SEED)

    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5,
        stratify=temp_lbl, random_state=SEED)

    return train_idx, val_idx, test_idx

# ── Module 1: Spectral Encoder ────────────────────────────────────────────────

class BidirectionalSSM(nn.Module):
    def __init__(self, n_bands, d_model=32):
        super().__init__()
        self.d_model   = d_model
        self.use_mamba = False
        try:
            from mamba_ssm import Mamba
            self.fwd_mamba = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            self.bwd_mamba = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            self.proj_in   = nn.Linear(1, d_model)
            self.use_mamba = True
            print('BidirectionalSSM: mamba_ssm.Mamba')
        except ImportError:
            self.bigru = nn.GRU(input_size=1, hidden_size=d_model,
                                num_layers=2, batch_first=True,
                                bidirectional=True, dropout=0.1)
            print('BidirectionalSSM: BiGRU fallback')

    def forward(self, x):
        seq = x.unsqueeze(-1)                        # (B, n_bands, 1)
        if self.use_mamba:
            seq = self.proj_in(seq)                  # (B, n_bands, d_model)
            fwd = self.fwd_mamba(seq)
            bwd = self.bwd_mamba(seq.flip(1)).flip(1)
            return torch.cat([fwd, bwd], dim=-1).mean(dim=1)  # (B, d_model*2)
        else:
            out, _ = self.bigru(seq)
            return out.mean(dim=1)                   # (B, d_model*2)


class NIRWedgeSeparator(nn.Module):
    def __init__(self, nir_start, n_bands, out_dim=16):
        super().__init__()
        self.nir_start = nir_start
        self.mlp = nn.Sequential(
            nn.Linear(n_bands - nir_start, 32),
            nn.LayerNorm(32), nn.GELU(),
            nn.Linear(32, out_dim)
        )

    def forward(self, x):
        return self.mlp(x[:, self.nir_start:])       # (B, out_dim)


class SpectralEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        ssm_dim  = 32
        nir_dim  = 16
        self.ssm = BidirectionalSSM(cfg.N_BANDS, d_model=ssm_dim)
        self.nir = NIRWedgeSeparator(cfg.NIR_START, cfg.N_BANDS, out_dim=nir_dim)
        self.proj = nn.Sequential(
            nn.Linear(ssm_dim * 2 + nir_dim, cfg.SPEC_DIM),
            nn.LayerNorm(cfg.SPEC_DIM), nn.GELU()
        )
        # M4 stub — unmixing head
        self.unmix_head = nn.Sequential(
            nn.Linear(cfg.SPEC_DIM, 16), nn.GELU(),
            nn.Linear(16, 2), nn.Softmax(dim=-1)
        )

    def forward(self, x):
        feat = self.proj(torch.cat([self.ssm(x), self.nir(x)], dim=-1))
        return feat, self.unmix_head(feat)

# ── Module 2: Dual-Scale Spatial Context ─────────────────────────────────────

class SpatialBranch(nn.Module):
    def __init__(self, n_bands, out_dim, patch_size):
        super().__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(1,  8, kernel_size=(7, 3, 3), padding=(3, 1, 1)),
            nn.BatchNorm3d(8),  nn.GELU(),
            nn.Conv3d(8, 16, kernel_size=(5, 3, 3), padding=(2, 1, 1)),
            nn.BatchNorm3d(16), nn.GELU(),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32), nn.GELU(),
        )
        self.spectral_pool = nn.AdaptiveAvgPool3d((1, patch_size, patch_size))
        self.spatial_pool  = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Linear(32, out_dim), nn.LayerNorm(out_dim), nn.GELU()
        )

    def forward(self, x):
        x = self.conv3d(x.unsqueeze(1))       # (B, 32, n_bands, S, S)
        x = self.spectral_pool(x).squeeze(2)  # (B, 32, S, S)
        x = self.spatial_pool(x).view(x.size(0), -1)  # (B, 32)
        return self.proj(x)                   # (B, out_dim)


class DualScaleSpatialContext(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Both branches output full SPAT_DIM
        self.local_branch  = SpatialBranch(cfg.N_BANDS, cfg.SPAT_DIM, cfg.LOCAL_SIZE)
        self.global_branch = SpatialBranch(cfg.N_BANDS, cfg.SPAT_DIM, cfg.GLOBAL_SIZE)
        # Scalar gate — single bottleneck, interpretable per class
        self.gate = nn.Sequential(
            nn.Linear(cfg.SPAT_DIM * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()   # (B, 1): 1=trust local, 0=trust global
        )

    def forward(self, local_patch, global_patch):
        F_l = self.local_branch(local_patch)             # (B, SPAT_DIM)
        F_g = self.global_branch(global_patch)           # (B, SPAT_DIM)
        g   = self.gate(torch.cat([F_l, F_g], dim=-1))  # (B, 1)
        return g * F_l + (1 - g) * F_g, g               # (B, SPAT_DIM), (B, 1)

# ── Full Hy-BiSSM ─────────────────────────────────────────────────────────────

class HyBiSSMNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.spectral_encoder = SpectralEncoder(cfg)
        self.spatial_context  = DualScaleSpatialContext(cfg)

        self.classifier = nn.Sequential(
            nn.Linear(cfg.SPEC_DIM + cfg.SPAT_DIM, 256),
            nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, cfg.N_CLASSES)
        )

        # M3 stub — (N_CLASSES, N_SCENES, SPEC_DIM)
        self.register_buffer('prototype_memory',
                             torch.zeros(cfg.N_CLASSES, 10, cfg.SPEC_DIM))

    def forward(self, local_patch, global_patch):
        B, C, H, W  = local_patch.shape
        center_spec = local_patch[:, :, H // 2, W // 2]          # (B, n_bands)
        spec_feat, unmix_out = self.spectral_encoder(center_spec)
        spat_feat,  gate_val = self.spatial_context(local_patch, global_patch)
        logits = self.classifier(torch.cat([spec_feat, spat_feat], dim=-1))
        return logits, spec_feat, unmix_out, gate_val

# ── Loss & Optimizer ─────────────────────────────────────────────────────────

class HyBiSSMLoss(nn.Module):
    """
    Compound loss.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg          = cfg
        self.ce           = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.road_cls     = cfg.CLASS_NAMES.index('Road')
        self.building_cls = cfg.CLASS_NAMES.index('Building')
        self.margin       = 1.0

    def contrastive_road_building(self, spec_feat, labels):
        mask_r = (labels == self.road_cls)
        mask_b = (labels == self.building_cls)
        if mask_r.sum() == 0 or mask_b.sum() == 0:
            return torch.tensor(0.0, device=spec_feat.device)
        r_proto = spec_feat[mask_r].mean(0, keepdim=True)
        b_proto = spec_feat[mask_b].mean(0, keepdim=True)
        return F.relu(self.margin - F.pairwise_distance(r_proto, b_proto)).mean()

    def forward(self, logits, labels, spec_feat, unmix_out,
                unmix_targets=None, morph_loss=None):
        L_CE = self.ce(logits, labels)

        L_contrast = torch.tensor(0.0, device=logits.device)
        if self.cfg.LAMBDA_CONTRAST > 0:
            L_contrast = self.contrastive_road_building(spec_feat, labels)

        L_unmix = torch.tensor(0.0, device=logits.device)
        if self.cfg.LAMBDA_UNMIX > 0 and unmix_targets is not None:
            L_unmix = F.mse_loss(unmix_out, unmix_targets)

        L_morph = morph_loss if morph_loss is not None \
                  else torch.tensor(0.0, device=logits.device)

        total = (L_CE
                 + self.cfg.LAMBDA_CONTRAST * L_contrast
                 + self.cfg.LAMBDA_UNMIX    * L_unmix
                 + self.cfg.LAMBDA_MORPH    * L_morph)
        return total, L_CE, L_contrast, L_unmix

# ── Epoch Runner Function ─────────────────────────────────────────────────────

def run_epoch(loader, model, criterion, optimizer=None, device=DEVICE):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = ce_loss_sum = 0.0
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    raw_m = model.module if hasattr(model, 'module') else model
    with ctx:
        for local_p, global_p, lbls, _ in loader:
            local_p  = local_p.to(device)
            global_p = global_p.to(device)
            lbls     = lbls.to(device)

            logits, spec_feat, unmix_out, gate_val = model(local_p, global_p)

            # DataParallel splits batch across GPUs; gather spec_feat for loss
            loss, l_ce, l_cont, l_unmix = criterion(
                logits, lbls, spec_feat, unmix_out)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(raw_m.parameters(), max_norm=1.0)
                optimizer.step()

            n = len(lbls)
            total_loss    += loss.item() * n
            ce_loss_sum   += l_ce.item() * n
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(lbls.cpu().numpy())

    oa    = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    cm    = confusion_matrix(all_labels, all_preds,
                             labels=list(range(CFG.N_CLASSES)))
    
    # Fix: Calculate AA robustly by only averaging over classes that have ground-truth samples
    class_counts = cm.sum(axis=1)
    recalls = cm.diagonal() / (class_counts + 1e-8)
    valid_classes = class_counts > 0
    aa = recalls[valid_classes].mean() if valid_classes.sum() > 0 else 0.0
    
    N     = len(all_labels)
    return total_loss / N, ce_loss_sum / N, oa, aa, kappa

# ── Main Guarded Script Execution ─────────────────────────────────────────────

if __name__ == '__main__':
    # Load all 10 scenes and build ConcatDatasets
    # Also keep per-scene references for per-scene evaluation and prototype extraction

    print('Loading scenes...')
    scenes = {}   # {scene_id: (image, labels)}
    all_train_ds, all_val_ds, all_test_ds = [], [], []

    # Store per-scene split indices for later prototype extraction
    scene_splits = {}   # {scene_id: (train_idx, val_idx, test_idx)}

    for sid in CFG.ALL_SCENES:
        image, labels = load_ohid1_scene(CFG, scene_idx=sid)
        scenes[sid]   = (image, labels)

        train_idx, val_idx, test_idx = make_splits(labels, CFG)
        scene_splits[sid] = (train_idx, val_idx, test_idx)

        all_train_ds.append(DualPatchDataset(image, labels, CFG, train_idx,
                                             scene_id=sid, augment=True))
        all_val_ds.append(DualPatchDataset(image, labels, CFG, val_idx,
                                           scene_id=sid, augment=False))
        all_test_ds.append(DualPatchDataset(image, labels, CFG, test_idx,
                                            scene_id=sid, augment=False))

    train_loader = DataLoader(ConcatDataset(all_train_ds),
                              batch_size=CFG.BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(ConcatDataset(all_val_ds),
                              batch_size=CFG.BATCH_SIZE * 2, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(ConcatDataset(all_test_ds),
                              batch_size=CFG.BATCH_SIZE * 2, shuffle=False,
                              num_workers=4, pin_memory=True)

    total_train = sum(len(d) for d in all_train_ds)
    total_val   = sum(len(d) for d in all_val_ds)
    total_test  = sum(len(d) for d in all_test_ds)
    print(f'\nTotal — Train: {total_train:,}  Val: {total_val:,}  Test: {total_test:,}')
    print(f'Steps per epoch: {len(train_loader):,}')

    # Build model, wrap in DataParallel if 2 GPUs available
    _model = HyBiSSMNet(CFG).to(DEVICE)
    if N_GPUS > 1:
        model = nn.DataParallel(_model)
        print(f'DataParallel across {N_GPUS} GPUs')
    else:
        model = _model

    total_params    = sum(p.numel() for p in _model.parameters())
    trainable_params = sum(p.numel() for p in _model.parameters() if p.requires_grad)
    print(f'Parameters : {total_params:,}  trainable: {trainable_params:,}')

    criterion = HyBiSSMLoss(CFG).to(DEVICE)

    # Access underlying model params regardless of DataParallel wrapper
    raw_model = model.module if N_GPUS > 1 else model
    optimizer = torch.optim.AdamW(raw_model.parameters(),
                                  lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG.EPOCHS, eta_min=1e-6)

    RESUME_FROM_CHECKPOINT = True   # set False for fresh training

    if RESUME_FROM_CHECKPOINT:
        # Load best checkpoint
        ckpt = torch.load(CFG.SAVE_PATH, map_location=DEVICE, weights_only=False)
        raw_model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        
        start_epoch = ckpt['epoch'] + 1
        best_val_oa = ckpt['val_oa']
        no_improve  = 0
        
        # Continue cosine schedule from where it left off
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CFG.EPOCHS, eta_min=1e-6,
            last_epoch=start_epoch - 2
        )
        print(f'Resumed from epoch {ckpt["epoch"]}  val OA={ckpt["val_oa"]:.4f}')
        print(f'Continuing from epoch {start_epoch} → {CFG.EPOCHS}')
    else:
        start_epoch = 1
        best_val_oa = 0.0
        no_improve  = 0

    history = {k: [] for k in
               ['train_loss','val_loss','train_oa','val_oa',
                'train_aa','val_aa','train_kappa','val_kappa']}

    # Fix: Only override best_val_oa and no_improve if not resuming
    if not RESUME_FROM_CHECKPOINT:
        best_val_oa = 0.0
        no_improve  = 0

    print(f'{"Epoch":>6}  {"Tr.Loss":>8}  {"Tr.OA":>7}  '
          f'{"Val.Loss":>8}  {"Val.OA":>7}  {"Val.AA":>7}  {"Val.K":>7}  {"LR":>9}')
    print('-' * 80)

    t0 = time.time()
    for epoch in range(start_epoch, CFG.EPOCHS + 1):
        tr = run_epoch(train_loader, model, criterion, optimizer)
        vl = run_epoch(val_loader,   model, criterion)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        for key, val in zip(
            ['train_loss','val_loss','train_oa','val_oa',
             'train_aa','val_aa','train_kappa','val_kappa'],
            [tr[0], vl[0], tr[2], vl[2], tr[3], vl[3], tr[4], vl[4]]):
            history[key].append(val)

        if epoch % 5 == 0 or epoch == 1:
            elapsed = (time.time() - t0) / 60
            print(f'{epoch:>6}  {tr[0]:>8.4f}  {tr[2]:>7.4f}  '
                  f'{vl[0]:>8.4f}  {vl[2]:>7.4f}  {vl[3]:>7.4f}  '
                  f'{vl[4]:>7.4f}  {lr:>9.2e}  [{elapsed:.0f}m]')

        if vl[2] > best_val_oa:
            best_val_oa = vl[2]
            no_improve  = 0
            cfg_dict = {k: v for k, v in CFG.__dict__.items()
                        if not k.startswith('__') and not callable(v)}
            
            # Fix: Ensure checkpoint parent directory exists before saving
            save_dir = os.path.dirname(CFG.SAVE_PATH)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                
            torch.save({
                'epoch':           epoch,
                'model_state':     raw_model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_oa':          vl[2],
                'val_aa':          vl[3],
                'val_kappa':       vl[4],
                'cfg':             cfg_dict,
            }, CFG.SAVE_PATH)
        else:
            no_improve += 1
            if no_improve >= CFG.PATIENCE:
                print(f'\nEarly stopping at epoch {epoch}.')
                break

    total_min = (time.time() - t0) / 60
    print(f'\nDone in {total_min:.1f} min  |  Best val OA: {best_val_oa:.4f}')

    ep = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(ep, history['train_loss'], label='Train', color='#534AB7')
    axes[0].plot(ep, history['val_loss'],   label='Val',   color='#D85A30')
    axes[0].set_title('Loss'); axes[0].set_xlabel('Epoch')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(ep, history['train_oa'], label='Train OA', color='#534AB7')
    axes[1].plot(ep, history['val_oa'],   label='Val OA',   color='#D85A30')
    axes[1].plot(ep, history['val_aa'],   label='Val AA',   color='#1D9E75', linestyle='--')
    axes[1].set_title('OA & AA'); axes[1].set_xlabel('Epoch')
    axes[1].set_ylim([0, 1]); axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(ep, history['train_kappa'], label='Train κ', color='#534AB7')
    axes[2].plot(ep, history['val_kappa'],   label='Val κ',   color='#D85A30')
    axes[2].set_title('Kappa'); axes[2].set_xlabel('Epoch')
    axes[2].set_ylim([0, 1]); axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.suptitle('Hy-BiSSM — All 10 Scenes', y=1.02, fontsize=13)
    plt.tight_layout()
    # Fix: Use local relative path for plotting instead of absolute Kaggle path
    plt.savefig('learning_curves.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Load best checkpoint
    ckpt = torch.load(CFG.SAVE_PATH, map_location=DEVICE)
    raw_model.load_state_dict(ckpt['model_state'])
    print(f"Checkpoint epoch {ckpt['epoch']}  val OA={ckpt['val_oa']:.4f}")

    # ── Overall test metrics ───────────────────────────────────────────────────────
    _, _, oa, aa, kap = run_epoch(test_loader, model, criterion)
    print(f'\nOverall Test — OA: {oa:.4f}  AA: {aa:.4f}  Kappa: {kap:.4f}')

    # ── Per-scene test metrics ────────────────────────────────────────────────────
    print('\nPer-scene breakdown:')
    print(f'{"Scene":>6}  {"OA":>7}  {"AA":>7}  {"Kappa":>7}  {"N_test":>8}')
    print('-' * 48)

    scene_metrics = {}
    for sid in CFG.ALL_SCENES:
        image, labels     = scenes[sid]
        _, val_idx, test_idx = scene_splits[sid]
        ds     = DualPatchDataset(image, labels, CFG, test_idx,
                                  scene_id=sid, augment=False)
        loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=2)
        _, _, s_oa, s_aa, s_kap = run_epoch(loader, model, criterion)
        scene_metrics[sid] = (s_oa, s_aa, s_kap)
        print(f'{sid:>6}  {s_oa:>7.4f}  {s_aa:>7.4f}  {s_kap:>7.4f}  {len(ds):>8,}')

    # ── Confusion matrix on full test set ─────────────────────────────────────────
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for lp, gp, lbls, _ in test_loader:
            logits, sf, _, _ = model(lp.to(DEVICE), gp.to(DEVICE))
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(lbls.numpy())

    cm      = confusion_matrix(all_labels, all_preds, labels=list(range(CFG.N_CLASSES)))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, data, fmt, title in zip(
        axes, [cm, cm_norm], ['d', '.2f'],
        ['Counts', 'Row-normalised recall']):
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=CFG.CLASS_NAMES,
                    yticklabels=CFG.CLASS_NAMES,
                    ax=ax, linewidths=0.5)
        ax.set_title(title)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    ri = CFG.CLASS_NAMES.index('Road')
    bi = CFG.CLASS_NAMES.index('Building')
    for ax in axes:
        ax.add_patch(plt.Rectangle((bi, ri), 1, 1, fill=False,
                                    edgecolor='red',    lw=2.5))
        ax.add_patch(plt.Rectangle((ri, bi), 1, 1, fill=False,
                                    edgecolor='orange', lw=2.5))

    plt.suptitle(f'All-Scene — OA={oa:.4f}  AA={aa:.4f}  κ={kap:.4f}',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    # Fix: Use local relative path for plotting instead of absolute Kaggle path
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Road→Building : {cm_norm[ri,bi]:.3f}  Building→Road : {cm_norm[bi,ri]:.3f}')

    print('Extracting prototypes for all 10 scenes...')
    raw_model.eval()
    all_prototypes = {}   # {scene_id: {class_id: tensor(SPEC_DIM)}}

    for sid in CFG.ALL_SCENES:
        image, labels = scenes[sid]
        rows, cols    = np.where(labels > 0)
        all_idx       = list(zip(rows.tolist(), cols.tolist()))
        ds     = DualPatchDataset(image, labels, CFG, all_idx,
                                  scene_id=sid, augment=False)
        loader = DataLoader(ds, batch_size=512, shuffle=False,
                            num_workers=2, pin_memory=True)

        feats, lbls = [], []
        with torch.no_grad():
            for lp, gp, l, _ in loader:
                _, sf, _, _ = model(lp.to(DEVICE), gp.to(DEVICE))
                feats.append(sf.cpu())
                lbls.append(l)

        feats = torch.cat(feats)
        lbls  = torch.cat(lbls).numpy()

        proto_dict = {}
        for c in range(CFG.N_CLASSES):
            mask = lbls == c
            proto_dict[c] = feats[mask].mean(0) if mask.sum() > 0 \
                            else torch.zeros(CFG.SPEC_DIM)
        all_prototypes[sid] = proto_dict
        print(f'  Scene {sid:>2} done')

    # Fix: Ensure prototype parent directory exists before saving
    proto_dir = os.path.dirname(CFG.PROTO_PATH)
    if proto_dir:
        os.makedirs(proto_dir, exist_ok=True)
        
    torch.save({
        'prototypes':    all_prototypes,
        'scene_metrics': scene_metrics,
        'overall':       {'oa': oa, 'aa': aa, 'kappa': kap},
        'epoch':         ckpt['epoch'],
    }, CFG.PROTO_PATH)

    print(f'\nSaved:')
    print(f'  {CFG.SAVE_PATH}')
    print(f'  {CFG.PROTO_PATH}')
    print(f'\nTraining complete — OA: {oa:.4f}  AA: {aa:.4f}  Kappa: {kap:.4f}')