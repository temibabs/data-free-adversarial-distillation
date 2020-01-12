import argparse

from data_loader import get_dataset
from solver import Solver


def str2bool(v: str):
    return v.lower() == 'true'


def main(config):

    if config.mode == 'train':
        train_dataset = get_dataset(config.batch_size,
                                    config.mode)  # MINST dataset

        solver = Solver(train_dataset, vars(config))
        # solver.train(train_dataset)

    elif config.mode == 'test':
        test_dataset = get_dataset(config.batch_size,
                                    config.mode)  # MINST dataset
        print('test_mode')
        solver = Solver(test_dataset, vars(config))
        solver.test(test_dataset)


if __name__ == '__main__':
    print("Initializing prototype parameters")

    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--lr', type=float, default=35e-1)

    # Training settings
    parser.add_argument('--train_teacher', type=str2bool, default='False')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--generator_steps', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--pretrained_model', type=str, default=None)

    # Misc
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default='true')

    # Path
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--log_path', type=str, default='./dagmm/logs')
    parser.add_argument('--teacher_model_path', type=str,
                        default='C:\\Users\\Temiloluwa.babalola'
                                '\\PycharmProjects\\data-free-adversarial'
                                '-distillation\\dagmm\\models\\teacher')
    parser.add_argument('--student_model_path', type=str,
                        default='C:\\Users\\Temiloluwa.babalola'
                                '\\PycharmProjects\\data-free-adversarial'
                                '-distillation\\dagmm\\models\\student')
    parser.add_argument('--generator_model_path', type=str,
                        default='C:\\Users\\Temiloluwa.babalola'
                                '\\PycharmProjects\\data-free-adversarial'
                                '-distillation\\dagmm\\models\\generator')

    # Step size
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=194)
    parser.add_argument('--model_save_step', type=int, default=194)

    configuration = parser.parse_args()

    args = vars(configuration)
    print('------------ Options -------------')
    for key, value in sorted(args.items()):
        print('%s: %s' % (str(key), str(value)))
    print('-------------- End ----------------')

    main(configuration)
