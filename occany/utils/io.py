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
            # build parameter name map for current optimizer params
            # name_map = {}
            # for n, p in img_encoder.named_parameters():
            #     name_map[id(p)] = f"encoder.{n}"
            # if raymap_encoder is not None:
            #     for n, p in raymap_encoder.named_parameters():
            #         name_map[id(p)] = f"raymap_encoder.{n}"
            # trainable_decoder = gen_decoder if gen_decoder is not None else decoder
            # decoder_prefix = "gen_decoder" if gen_decoder is not None else "decoder"
            # for n, p in trainable_decoder.named_parameters():
            #     name_map[id(p)] = f"{decoder_prefix}.{n}"

            # # try:
            # print("[DEBUG] checkpoint optimizer groups:", len(optim_state_dict.get('param_groups', [])))
            # print("[DEBUG] current optimizer groups:", len(optimizer.param_groups))
            # cp_groups = optim_state_dict.get('param_groups', [])
            # if cp_groups:
            #     cp_lens = [len(g.get('params', [])) for g in cp_groups]
            #     cur_lens = [len(g.get('params', [])) for g in optimizer.param_groups]
            #     print("[DEBUG] checkpoint group param lens:", cp_lens)
            #     print("[DEBUG] current group param lens:", cur_lens)
            #     print("[DEBUG] first checkpoint group keys:", list(cp_groups[0].keys()))
            #     print("[DEBUG] first current group keys:", list(optimizer.param_groups[0].keys()))
            # else:
            #     cp_lens = []
            #     cur_lens = []
            # for gi, g in enumerate(optimizer.param_groups):
            #     curr_names = [name_map.get(id(p), f"<unnamed_param_{i}>") for i, p in enumerate(g.get('params', []))]
            #     print(f"[DEBUG] current group {gi} param names:", curr_names)
            #     if gi < len(cp_lens):
            #         extra = cur_lens[gi] - cp_lens[gi]
            #         if extra > 0:
            #             new_names = curr_names[-extra:]
            #             print(f"[DEBUG] extra params in current group {gi} (not in checkpoint):", new_names)
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
