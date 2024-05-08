# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from loguru import logger
import os
from timm.utils import get_state_dict
import torch as t


def save_checkpoint(epoch, model, extras=None, is_best=None, name=None, output_dir='.', optimizer=None, model_ema=None):
    """Save a pyTorch training checkpoint
    Args:
        epoch: current epoch number
        model: a pyTorch model
        extras: optional dict with additional user-defined data to be saved in the checkpoint.
            Will be saved under the key 'extras'
        is_best: If true, will save a copy of the checkpoint with the suffix 'best'
        name: the name of the checkpoint file
        output_dir: directory in which to save the checkpoint
    """
    if not os.path.isdir(output_dir):
        raise IOError('Checkpoint directory does not exist at', os.path.abspath(dir))

    if extras is None:
        extras = {}
    if not isinstance(extras, dict):
        raise TypeError('extras must be either a dict or None')

    filename = 'checkpoint.pth.tar' if name is None else name + '_checkpoint.pth.tar'
    filepath = os.path.join(output_dir, filename)
    filename_best = 'best.pth.tar' if name is None else name + '_best.pth.tar'
    filepath_best = os.path.join(output_dir, filename_best)

    checkpoint = {
        'epoch': epoch,
        'state_dict': get_state_dict(model),
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'extras': extras,
        'model_ema': get_state_dict(model_ema)
    }

    msg = 'Saving checkpoint to:\n'
    msg += '             Current: %s\n' % filepath
    t.save(checkpoint, filepath)
    if is_best:
        msg += '                Best: %s\n' % filepath_best
        t.save(checkpoint, filepath_best)
    logger.info(msg)


def load_checkpoint(model, chkp_file, strict=True, lean=False, optimizer=None):
    """Load a pyTorch training checkpoint.
    Args:
        model: the pyTorch model to which we will load the parameters.  You can
        specify model=None if the checkpoint contains enough metadata to infer
        the model.  The order of the arguments is misleading and clunky, and is
        kept this way for backward compatibility.
        chkp_file: the checkpoint file
        lean: if set, read into model only 'state_dict' field
        model_device [str]: if set, call model.to($model_device)
                This should be set to either 'cpu' or 'cuda'.
    :returns: updated model, optimizer, start_epoch
    """

    if not os.path.isfile(chkp_file):
        raise NotImplementedError

    checkpoint = t.load(chkp_file, map_location=lambda storage, loc: storage)

    if 'state_dict' not in checkpoint:
        raise ValueError('Checkpoint must contain model parameters')

    extras = checkpoint.get('extras', None)

    checkpoint_epoch = checkpoint.get('epoch', None)
    start_epoch = checkpoint_epoch + 1 if checkpoint_epoch is not None else 0

    new_state_dict = {}
    for k in checkpoint['state_dict'].keys():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = checkpoint['state_dict'][k]
    
    ema = checkpoint.get('model_ema', None)

    if len(new_state_dict) == 0:
        checkpoint['state_dict'] = checkpoint['model_ema']
        print('loading cpt from ema_model...')
    else:
        assert len(checkpoint['state_dict']) == len(new_state_dict)
        
        if ema is not None:
            checkpoint['state_dict'] = checkpoint['model_ema']
            print('loading cpt from ema_model (reset)...')
        else:
            checkpoint['state_dict'] = new_state_dict
            print('loading cpt from training model...')

    anomalous_keys = model.load_state_dict(checkpoint['state_dict'], strict)

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    if strict:
        if anomalous_keys:
            missing_keys, unexpected_keys = anomalous_keys
            if unexpected_keys:
                logger.warning("The loaded checkpoint (%s) contains %d unexpected state keys" %
                            (chkp_file, len(unexpected_keys)))
            if missing_keys:
                print(missing_keys)
                raise ValueError("The loaded checkpoint (%s) is missing %d state keys" %
                                (chkp_file, len(missing_keys)))
            

    model.cuda()

    if lean:
        logger.info("Loaded checkpoint %s model (next epoch %d) from %s", arch, 0, chkp_file)
        return model, 0, None
    else:
        logger.info("Loaded checkpoint %s model (next epoch %d) from %s", arch, start_epoch, chkp_file)
        return model, start_epoch, extras
