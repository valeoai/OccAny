import torch
from pathlib import Path

from occany.utils.checkpoint_io import save_on_master

def save_model(args, epoch, img_encoder, raymap_encoder, decoder, optimizer, loss_scaler, fname=None, gen_decoder=None):
    output_dir = Path(args.output_dir)
    if fname is None:
        fname = str(epoch)
    checkpoint_path = output_dir / ('checkpoint-%s.pth' % fname)
    optim_state_dict = optimizer.state_dict()
    to_save = {
        'encoder': img_encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'optimizer': optim_state_dict,
        'scaler': loss_scaler.state_dict(),
        'args': args,
        'epoch': epoch,
    }
    if raymap_encoder is not None:
        to_save['raymap_encoder'] = raymap_encoder.state_dict()
    if gen_decoder is not None:
        to_save['gen_decoder'] = gen_decoder.state_dict()
    
    print(f'>> Saving model to {checkpoint_path} ...')
    print('   - Saving: encoder, decoder, optimizer, scaler')
    if raymap_encoder is not None:
        print('   - Saving: raymap_encoder')
    if gen_decoder is not None:
        print('   - Saving: gen_decoder')
    save_on_master(to_save, checkpoint_path)


def load_model(args, chkpt_path, img_encoder, raymap_encoder, decoder, optimizer, loss_scaler, gen_decoder=None):
    args.start_epoch = 0
    if chkpt_path is not None:
        checkpoint = torch.load(chkpt_path, map_location='cpu', weights_only=False)

        print("Resume checkpoint %s" % chkpt_path)
        print('   - Loading: encoder')
        img_encoder.load_state_dict(checkpoint['encoder'], strict=False)
        if raymap_encoder is not None:
            print('   - Loading: raymap_encoder')
            raymap_encoder.load_state_dict(checkpoint['raymap_encoder'], strict=False)
        print('   - Loading: decoder')
        decoder.load_state_dict(checkpoint['decoder'], strict=False)
        if gen_decoder is not None and 'gen_decoder' in checkpoint:
            print('   - Loading: gen_decoder')
            gen_decoder.load_state_dict(checkpoint['gen_decoder'], strict=False)
        elif gen_decoder is not None:
            print('   - Warning: gen_decoder not found in checkpoint')
        args.start_epoch = checkpoint['epoch'] + 1
        
        if 'optimizer' in checkpoint:
            optim_state_dict = checkpoint['optimizer']
       

            try:
                optimizer.load_state_dict(optim_state_dict)
                if 'scaler' in checkpoint:
                    loss_scaler.load_state_dict(checkpoint['scaler'])
                print("With optim & sched! start_epoch={:d}".format(args.start_epoch), end='')
            except ValueError as e:
                print(f"Warning: Could not load optimizer state: {e}")
                print("Starting with fresh optimizer state. Resetting to epoch 0.")
                args.start_epoch = 0
        else:
            print("No optimizer state found in checkpoint. Starting with fresh optimizer state.")
