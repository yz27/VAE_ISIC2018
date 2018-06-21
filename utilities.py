import time
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def trainVAE(train_loader, model, criterion, optimizer, epoch, writer, args):
    """
    Iterate through the train data and perform optimization
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_avg = AverageMeter()
    recon_loss_avg = AverageMeter()
    kl_loss_avg = AverageMeter()


    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.cuda:
            input = input.cuda()

        recon_batch, mu, logvar = model(input)
        recon_loss, kl_loss = criterion(recon_batch, input, mu, logvar)
        loss = recon_loss + kl_loss

        # record loss
        recon_loss_avg.update(recon_loss.item(), input.size(0))
        kl_loss_avg.update(kl_loss.item(), input.size(0))
        loss_avg.update(loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'recon_loss {recon_loss.val:.4f} ({recon_loss.avg:.4f})\t'
                  'kl_loss {kl_loss.val:.4f} ({kl_loss.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, recon_loss=recon_loss_avg, kl_loss=kl_loss_avg,
                   loss=loss_avg))

        # tensorboard logging
        writer.add_scalar('train_loss', loss_avg.avg)
        writer.add_scalar('recon_loss', recon_loss_avg.avg)
        writer.add_scalar('kl_loss', kl_loss_avg.avg)


def validateVAE(val_loader, model, criterion, args):
    """
    iterate through the validate set and output the accuracy
    """
    batch_time = AverageMeter()
    loss_avg = AverageMeter()
    recon_loss_avg = AverageMeter()
    kl_loss_avg = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        if args.cuda:
            target = target.cuda()
            input = input.cuda()

        # compute output
        recon_batch, mu, logvar = model(input)
        recon_loss, kl_loss = criterion(recon_batch, input, mu, logvar)
        loss = recon_loss + kl_loss

        # measure accuracy and record loss
        recon_loss_avg.update(recon_loss.item(), input.size(0))
        kl_loss_avg.update(kl_loss.item(), input.size(0))
        loss_avg.update(loss.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'recon_loss {recon_loss.val:.4f} ({recon_loss.avg:.4f})\t'
                  'kl_loss {kl_loss.val:.4f} ({kl_loss.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   i, len(val_loader), batch_time=batch_time, recon_loss=recon_loss_avg,
                   kl_loss= kl_loss_avg, loss=loss_avg))
    return loss_avg.avg


def load_opt(args, base_params, top_params):
    """
    Load the optimizer and lr scheduler. If resume and not reset opt,
    load from checkpoint. Only use adam now with default betas.
    :param base_params: a list of CNN base parameters
    :param top_params: a list of fc parameters
    :return: opt, scheduler
    """
    if args.fine_tune:
        opt_specs = [
            {'params' : base_params, 'lr' : args.base_lr},
            {'params' : top_params, 'lr' : args.lr},
        ]
    else:
        all_params = base_params + top_params
        opt_specs = [{'params' : all_params, 'lr' : args.lr}]
    opt = Adam(opt_specs, weight_decay=args.l2_decay, betas=(0.9, 0.999))
    scheduler = MultiStepLR(opt, milestones=args.schedule, gamma=args.lr_decay)

    # load from checkpoint only if we don't reset optimizer
    if args.resume is not None and args.reset_opt is False:
        checkpoint = torch.load(args.resume)
        opt.load_state_dict(checkpoint['optimizer'])
    return opt, scheduler