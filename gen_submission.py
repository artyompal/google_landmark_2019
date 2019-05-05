#!/usr/bin/python3.6
''' Trains a model or infers predictions. '''


def generate_submission(val_loader: Any, test_loader: Any, model: Any,
                        label_encoder: Any, epoch: int, model_path: Any) -> np.ndarray:
    sample_sub = pd.read_csv('../data/recognition_sample_submission.csv')

    predicts, confs, _ = inference(test_loader, model)
    predicts, confs = predicts.cpu().numpy(), confs.cpu().numpy()

    labels = [label_encoder.inverse_transform(pred) for pred in predicts]
    print('labels')
    print(np.array(labels))
    print('confs')
    print(np.array(confs))

    sub = test_loader.dataset.df
    def concat(label, conf):
        return ' '.join([f'{L} {c}' for L, c in zip(label, conf)])
    sub['landmarks'] = [concat(label, conf) for label, conf in zip(labels, confs)]

    sample_sub = sample_sub.set_index('id')
    sub = sub.set_index('id')
    sample_sub.update(sub)

    sample_sub.to_csv(f'../submissions/{os.path.basename(model_path)[:-4]}.csv')

def run() -> float:
    np.random.seed(0)
    model_dir = config.experiment_dir

    logger.info('=' * 50)
    # logger.info(f'hyperparameters: {params}')

    train_loader, val_loader, test_loader, label_encoder = load_data(args.fold)
    model = create_model()

    optimizer = get_optimizer(config, model.parameters())
    lr_scheduler = get_scheduler(config, optimizer)
    criterion = get_loss(config)

    if args.weights is None:
        last_epoch = 0
        logger.info(f'training will start from epoch {last_epoch+1}')
    else:
        last_checkpoint = torch.load(args.weights)
        assert last_checkpoint['arch'] == config.model.arch
        model.load_state_dict(last_checkpoint['state_dict'])
        optimizer.load_state_dict(last_checkpoint['optimizer'])
        logger.info(f'checkpoint {args.weights} was loaded.')

        last_epoch = last_checkpoint['epoch']
        logger.info(f'loaded the model from epoch {last_epoch}')

        if args.lr_override != 0:
            set_lr(optimizer, float(args.lr_override))
        elif 'lr' in config.scheduler.params:
            set_lr(optimizer, config.scheduler.params.lr)

    if args.predict:
        print('inference mode')
        generate_submission(val_loader, test_loader, model, label_encoder,
                            last_epoch, args.weights)
        sys.exit(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='model configuration file (YAML)', type=str, required=True)
    parser.add_argument('--weights', help='model to resume training', type=str)
    parser.add_argument('--fold', help='fold number', type=int, default=0)
    parser.add_argument('--predict', help='model to resume training', action='store_true')
    parser.add_argument('--show_summary', help='show model summary', action='store_true')
    parser.add_argument('--lr_override', help='override learning rate', type=float, default=0)
    args = parser.parse_args()

    config = parse_config.load(args.config, args)

    if not os.path.exists(config.experiment_dir):
        os.makedirs(config.experiment_dir)

    log_filename = 'log_training.txt' if not args.predict else 'log_predict.txt'
    logger = create_logger(os.path.join(config.experiment_dir, log_filename))
    run()
