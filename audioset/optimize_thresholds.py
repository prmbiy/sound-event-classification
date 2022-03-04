import os
from sklearn import metrics
from utils import configureTorchDevice
import argparse
import time
import pickle
import numpy as np
from config import workspace, num_frames, num_classes, feature_type, permutation, seed

class HyperParamsOptimizer(object):
    def __init__(self, score_calculator, save_dict, learning_rate=1e-2, epochs=100,
                 step=0.01, max_search=5):
        """Hyper parameters optimizer. Parameters are optimized using gradient
        descend methods by using the numerically calculated graident:
        gradient: f(x + h) - f(x) / (h)
        Args:
          score_calculator: object. See ScoreCalculatorExample in example.py as
              an example.
          learning_rate: float
          epochs: int
          step: float, equals h for calculating gradients
          max_search: int, if plateaued, then search for at most max_search times
        """

        self.score_calculator = score_calculator
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.optimizer = Adam()
        self.optimizer.alpha = learning_rate
        self.step = step
        self.max_search = max_search
        self.save_dict = save_dict

    def do_optimize(self, init_params):
        print('Optimizing hyper parameters ...')
        print('learning rate: {:.3f}, total epochs: {}'.format(
            self.learning_rate, self.epochs))

        params = init_params.copy()

        for i in range(self.epochs):
            t1 = time.time()
            (score, grads) = self.calculate_gradients(params)
            grads = [-e for e in grads]
            params = self.optimizer.GetNewParams(params, grads)
            self.save_dict[i] = {'thresholds': params, 'score': score}
            print('    Hyper parameters: {}, score: {:.4f}'.format(
                [round(param, 4) for param in params], score))
            print('    Epoch: {}, Time: {:.4f} s'.format(i, time.time() - t1))

        return score, params, self.save_dict

    def calculate_gradients(self, params):
        """Calculate gradient of thresholds numerically.
        Args:
          y_true: (N, (optional)frames_num], classes_num)
          output: (N, (optional)[frames_num], classes_num)
          thresholds: (classes_num,), initial thresholds
          average: 'micro' | 'macro'
        Returns:
          grads: vector
        """
        score = self.score_calculator(params)
        step = self.step
        grads = []

        for k, param in enumerate(params):
            new_params = params.copy()
            cnt = 0
            while cnt < self.max_search:
                cnt += 1
                new_params[k] += self.step
                new_score = self.score_calculator(new_params)

                if new_score != score:
                    break

            grad = (new_score - score) / (step * cnt)
            grads.append(grad)

        return score, grads


class Base(object):
    def _reset_memory(self, memory):
        for i1 in range(len(memory)):
            memory[i1] = np.zeros(memory[i1].shape)


class Adam(Base):
    def __init__(self):
        self.ms = []
        self.vs = []
        self.alpha = 1e-3
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.iter = 0

    def GetNewParams(self, params, gparams):
        if not self.ms:
            for param in params:
                self.ms += [np.zeros_like(param)]
                self.vs += [np.zeros_like(param)]

        # fast adam, faster than origin adam
        self.iter += 1
        new_params = []
        alpha_t = self.alpha * \
            np.sqrt(1 - np.power(self.beta2, self.iter)) / \
            (1 - np.power(self.beta1, self.iter))
        for i1 in range(len(params)):
            self.ms[i1] = self.beta1 * self.ms[i1] + \
                (1 - self.beta1) * gparams[i1]
            self.vs[i1] = self.beta2 * self.vs[i1] + \
                (1 - self.beta2) * np.square(gparams[i1])
            new_params += [params[i1] - alpha_t *
                           self.ms[i1] / (np.sqrt(self.vs[i1] + self.eps))]

        return new_params

    def reset(self):
        self._reset_memory(self.ms)
        self._reset_memory(self.vs)
        self.epoch = 1

def calculate_precision_recall_f1(y_true, output, thresholds, average='micro'):
    """Calculate precision, recall, F1."""
    if y_true.ndim == 3:
        (N, T, F) = y_true.shape
        y_true = y_true.reshape((N * T, F))
        output = output.reshape((N * T, F))

    classes_num = y_true.shape[-1]
    binarized_output = np.zeros_like(output)

    for k in range(classes_num):
        binarized_output[:, k] = (np.sign(output[:, k] - thresholds[k]) + 1) // 2

    if average == 'micro':
        precision = metrics.precision_score(y_true.flatten(), binarized_output.flatten())
        recall = metrics.recall_score(y_true.flatten(), binarized_output.flatten())
        f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1
    
    else:
        raise Exception('Incorrect argument!')

class AudioTaggingScoreCalculator(object):
    def __init__(self, prediction_path):
        """Used to calculate score (such as F1) given prediction, target and hyper parameters. 
        """

    def __call__(self, params):
        """Use hyper parameters to threshold prediction to obtain output.
        Then, the scores are calculated between output and target.
        """
        (precision, recall, f1) = calculate_precision_recall_f1(
            self.output_dict['target'], self.output_dict['clipwise_output'],
            thresholds=params)

        return f1


def optimize_at_thresholds(args):
    """Calculate audio tagging metrics with optimized thresholds.

    Args:
      dataset_dir: str
      workspace: str
      filename: str
      holdout_fold: '1'
      model_type: str, e.g., 'Cnn_9layers_Gru_FrameAtt'
      loss_type: str, e.g., 'clip_bce'
      augmentation: str, e.g., 'mixup'
      batch_size: int
      iteration: int
    """

    # Arugments & parameters
    classes_num = num_classes
    # Paths
    if data_type == 'test':
        reference_csv_path = os.path.join(dataset_dir, 'metadata',
                                          'groundtruth_strong_label_testing_set.csv')

    prediction_path = os.path.join(workspace, 'predictions',
                                   '{}'.format(filename), 'holdout_fold={}'.format(
                                       holdout_fold),
                                   'model_type={}'.format(
                                       model_type), 'loss_type={}'.format(loss_type),
                                   'augmentation={}'.format(
                                       augmentation), 'batch_size={}'.format(batch_size),
                                   '{}_iterations.prediction.{}.pkl'.format(iteration, data_type))

    opt_thresholds_path = os.path.join(workspace, 'opt_thresholds',
                                       '{}'.format(filename), 'holdout_fold={}'.format(
                                           holdout_fold),
                                       'model_type={}'.format(
                                           model_type), 'loss_type={}'.format(loss_type),
                                       'augmentation={}'.format(
                                           augmentation), 'batch_size={}'.format(batch_size),
                                       '{}_iterations.at.{}.pkl'.format(iteration, data_type))
    create_folder(os.path.dirname(opt_thresholds_path))

    # Score calculator
    score_calculator = AudioTaggingScoreCalculator(prediction_path)

    # Thresholds optimizer
    hyper_params_opt = HyperParamsOptimizer(
        score_calculator, learning_rate=1e-2, epochs=100)

    # Initialize thresholds
    init_params = [0.3] * classes_num
    score_no_opt = score_calculator(init_params)

    # Optimize thresholds
    (opt_score, opt_params) = hyper_params_opt.do_optimize(init_params=init_params)

    print('\n------ Optimized thresholds ------')
    print(np.around(opt_params, decimals=4))

    print('\n------ Without optimized thresholds ------')
    print('Score: {:.3f}'.format(score_no_opt))

    print('\n------ With optimized thresholds ------')
    print('Score: {:.3f}'.format(opt_score))

    # Write out optimized thresholds
    pickle.dump(opt_params, open(opt_thresholds_path, 'wb'))
    print('\nSave optimized thresholds to {}'.format(opt_thresholds_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')

    parser.add_argument('-w', '--workspace', type=str, default=workspace)
    parser.add_argument('-f', '--feature_type', type=str, default=feature_type)
    parser.add_argument('-n', '--num_frames', type=int, default=num_frames)
    parser.add_argument('-p', '--permutation', type=int,
                        nargs='+', default=permutation)
    parser.add_argument('-s', '--seed', type=int, default=seed)

    args = parser.parse_args()
    optimize_at_thresholds(args)
