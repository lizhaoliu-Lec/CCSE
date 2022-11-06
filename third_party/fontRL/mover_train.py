from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torchvision.transforms import GaussianBlur
from tqdm import tqdm

from DRL.actor import *
from DRL.mover import *
from env import *
from mover_dataset import load_mover_weights, FontDataset, bbox_validation, augmentation
from utils.tensorboard import TensorBoard

np.random.seed(1234)
torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)
random.seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument('--font', default='2', type=str)
parser.add_argument('--run-id', default='1', type=str)
parser.add_argument('--depth', default=34, type=int)
parser.add_argument('--pretrained', default=None, type=str)
parser.add_argument('--d2', action='store_true')
parser.add_argument('--bs', default=96, type=int)
parser.add_argument('--val-bs', default=512, type=int)
parser.add_argument('--actor-path', required=True, type=str)
parser.add_argument('--num-val', default=-1, type=int)
parser.add_argument('--optim', default='adam', type=str)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--wd', default=0., type=float)
parser.add_argument('--step', default=100, type=int)
parser.add_argument('--f-bn', action="store_true")
parser.add_argument('--dropout', default=0, type=float)
parser.add_argument('--aug-prob', default=0., type=float)
args = parser.parse_args()
font = args.font
actor_path = args.actor_path
optim = args.optim
lr = args.lr
wd = args.wd
decay_step = args.step
momentum = args.momentum
f_bn = args.f_bn
t_print("Freezing BN: {}".format(f_bn))
dropout = args.dropout
t_print("Using dropout: {}".format(dropout))
aug_prob = args.aug_prob
t_print("Using aug_prob: {}".format(aug_prob))
assert optim in ['sgd', 'adam'], 'found unsupported optimizer {}'.format(optim)
if optim == 'sgd':
    t_print("Using optimizer {} with lr {}, wd {}, momentum {}".format(optim, lr, wd, momentum))
else:
    t_print("Using optimizer {} with lr {}, wd {}".format(optim, lr, wd))
t_print("Decrease lr to 0.1 * lr every {} epoch".format(decay_step))
assert os.path.exists(actor_path), 'actor_path: {} not exist'.format(actor_path)
num_val = args.num_val
t_print("Using num_val={}".format(num_val))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
width = 128

channels = 3
maxstep = 1

actor = ResNet(2 * channels + 3, 18, 50)
t_print("Loading pretrained actor path from {}...".format(actor_path))
actor.load_state_dict(torch.load(actor_path))
actor = actor.to(device).eval()
bbox_net_depth = args.depth
t_print("Using ResNet{} for BBoxNet...".format(bbox_net_depth))
mover = ResNet_mover(3 * channels, bbox_net_depth, 3, dropout=dropout)
mover = load_mover_weights(mover, args.pretrained, args.d2)
if f_bn:
    t_print("Freezing BN in mover...")
    freeze_bn(mover)
mover = mover.to(device).train()

run_id = args.run_id
base_dir = 'model/{}/Paint-run-{}/mover'.format(font, run_id)
t_print("Saving ckpt and tensorboard to {}".format(base_dir))
if not os.path.exists(base_dir):
    os.makedirs(base_dir)
writer = TensorBoard(base_dir.format(font))

T = torch.ones([1, 1, width, width], dtype=torch.float32).to(device)
coord = torch.zeros([1, 2, width, width]).float().to(device)
coord[0, 0, :, :] = torch.linspace(-1, 1, 128)
coord[0, 1, :, :] = torch.linspace(-1, 1, 128)[..., None]
T = torch.ones([1, 1, width, width], dtype=torch.float32).to(device)

root1 = 'mean'
root2 = font
t_print(root2)

bs = args.bs
val_bs = args.val_bs
t_print("Using bs={}".format(bs))
t_print("Using val_bs={}".format(val_bs))

if optim == 'adam':
    mover_optim = Adam(mover.parameters(), lr=lr, weight_decay=wd)
else:
    mover_optim = SGD(mover.parameters(), lr=lr, momentum=momentum, weight_decay=wd)

t_print("Loading training dataset...")

train_dataset = FontDataset(img_path='data/font_concat/{}'.format(root2),
                            point_src_path='data/font_10_stroke/{}_point.npy'.format(root1),
                            point_tgt_path='data/font_10_stroke/{}_point.npy'.format(root2),
                            num_limit=-1,
                            train=True)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True, num_workers=8, drop_last=True)

t_print("Training dataset len: {}".format(len(train_dataset)))

val_dataset = FontDataset(img_path='data/font_concat/{}'.format(root2),
                          point_src_path='data/font_10_stroke/{}_point.npy'.format(root1),
                          point_tgt_path='data/font_10_stroke/{}_point.npy'.format(root2),
                          num_limit=num_val)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=val_bs, shuffle=False, num_workers=8)

global_step = 0
for epoch in range(150):
    mover.train()
    t_print("Train, epoch: {}, global_step: {}, begin training".format(epoch, global_step))
    if (epoch + 1) % decay_step == 0:
        t_print("Train, epoch: {}, global_step: {}, lower lr to 1e-5".format(epoch, global_step))
        if optim == 'adam':
            mover_optim = Adam(mover.parameters(), lr=lr / 10, weight_decay=wd)
        else:
            mover_optim = SGD(mover.parameters(), lr=lr / 10, momentum=momentum, weight_decay=wd)

    cnt = 0

    for src, ref, canvas, point, COL, img_id, stroke_id, add, tgt, ref_point, tgt_point in tqdm(train_dataloader):
        step = T.float() * 0 / maxstep

        # move all variable to cuda
        ref = ref.to(device)
        canvas = canvas.to(device)
        point = point.to(device)
        COL = COL.to(device)
        ref_point = ref_point.to(device)
        tgt_point = tgt_point.to(device)

        # print("===> canvas.size(): {}".format(canvas.size()))
        # print("===> ref.size(): {}".format(ref.size()))
        # print("===> point.size(): {}".format(point.size()))
        # print("===> tgt_point.size(): {}".format(tgt_point.size()))
        # print("===> COL.size(): {}".format(COL.size()))
        # print("===> ref_point.size(): {}".format(ref_point.size()))

        with torch.no_grad():
            state = torch.cat((canvas, ref, step.repeat(bs, 1, 1, 1), coord.repeat(bs, 1, 1, 1)), 1)
            action1 = actor(state.float())
            canvas1, point = decode(point, action1, False, ref_point)
            canvas1 = canvas1 / 255. * COL[:, :, None, None] / 255.

            state = torch.cat((canvas1, canvas, ref), 1)

            # perform augmentation here
            if random.random() < aug_prob:
                state = GaussianBlur(kernel_size=5)(state)

        action = mover(state)

        maxx = tgt_point[:, :, :1].max(1)[0]
        minx = tgt_point[:, :, :1].min(1)[0]
        maxy = tgt_point[:, :, 1:].max(1)[0]
        miny = tgt_point[:, :, 1:].min(1)[0]
        mxy = torch.cat(((maxx + minx) / 2, (maxy + miny) / 2), 1)
        dxy = (tgt_point - mxy[:, None, :]).max(1)[0].max(1)[0]

        loss2 = torch.abs(action[:, 0] - dxy).mean()
        loss3 = torch.abs(2 * action[:, 1] - 1 - mxy[:, 0]).mean()
        loss4 = torch.abs(2 * action[:, 2] - 1 - mxy[:, 1]).mean()

        loss = loss2 + loss3 + loss4

        mover_optim.zero_grad()
        loss.backward()
        # print("===> type(mover): {}".format(type(mover)))
        # print("===> type(mover.conv1): {}".format(type(mover.conv1)))
        # print("===> type(mover.conv1.weight): {}".format(type(mover.conv1.weight)))
        # print("===> type(mover.conv1.weight.grad): {}".format(type(mover.conv1.weight.grad)))
        # print("===> mover.grad.mean: {}".format(torch.mean(mover.conv1.weight.grad)))
        mover_optim.step()

        if global_step % 100 == 0:
            # t_print(epoch, global_step, loss.item(), loss2.item(), loss3.item(), loss4.item())
            t_print(
                "Train, epoch: {}, global_step: {}, loss: {:.4f}, loss2: {:.4f}, loss3: {:.4f}, loss4: {:.4f}".format(
                    epoch, global_step, loss.item(), loss2.item(), loss3.item(), loss4.item()
                ))
            writer.add_scalar('train/loss', loss, global_step)
            writer.add_scalar('train/loss2', loss2, global_step)
            writer.add_scalar('train/loss3', loss3, global_step)
            writer.add_scalar('train/loss4', loss4, global_step)
            torch.cuda.empty_cache()

        global_step += 1

    t_print("Train, epoch: {}, global_step: {}, complete training".format(epoch, global_step))

    # evaluate IoU and L1 Loss here
    t_print("Evaluating IoU and L1 Loss...")
    mover.eval()
    IoU, L1_Loss = bbox_validation(val_dataset=val_dataset,
                                   val_dataloader=val_dataloader,
                                   actor=actor,
                                   mover=mover,
                                   device=device,
                                   maxstep=maxstep,
                                   width=width)
    t_print("Evaluation done!")
    t_print("Val, epoch: {}, IoU: {:.4f}, L1_Loss: {:.4f}".format(epoch, IoU, L1_Loss))
    writer.add_scalar('val/IoU', IoU, epoch)
    writer.add_scalar('val/L1_Loss', L1_Loss, epoch)

    # save per epoch
    mover.cpu()
    save_dir = os.path.join(base_dir, 'mover-{}.pkl'.format(epoch))
    torch.save(mover.state_dict(), save_dir)
    mover.to(device)
    mover.train()
