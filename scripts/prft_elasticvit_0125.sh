model_path = args.onnx_path
    supernet_config_dict = toml.load(args.supernet_config_path)
    
    onnx_model = onnx.load(model_path)