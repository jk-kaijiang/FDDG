
import itertools
import numpy as np

def get_test_records(records):

    return records.filter(lambda r: len(r['args']['test_envs']) == 1)

class SelectionMethod:


    def __init__(self):
        raise TypeError

    @classmethod
    def run_acc(self, run_records):

        raise NotImplementedError

    @classmethod
    def hparams_accs(self, records):

        return (records.group('args.hparams_seed')
            .map(lambda _, run_records:
                (
                    self.run_acc(run_records),
                    run_records
                )
            ).filter(lambda x: x[0] is not None)
            .sorted(key=lambda x: x[0]['val_acc'])[::-1]
        )

    @classmethod
    def sweep_acc(self, records):

        _hparams_accs = self.hparams_accs(records)
        if len(_hparams_accs):
            return _hparams_accs[0][0]['test_acc']
        else:
            return None

class OracleSelectionMethod(SelectionMethod):

    name = "test-domain validation set (oracle)"

    @classmethod
    def run_acc(self, run_records):
        run_records = run_records.filter(lambda r:
            len(r['args']['test_envs']) == 1)
        if not len(run_records):
            return None
        test_env = run_records[0]['args']['test_envs'][0]
        test_out_acc_key = 'env{}_out_acc'.format(test_env)
        test_in_acc_key = 'env{}_in_acc'.format(test_env)
        chosen_record = run_records.sorted(lambda r: r['step'])[-1]
        return {
            'val_acc':  chosen_record[test_out_acc_key],
            'test_acc': chosen_record[test_in_acc_key]
        }

class IIDAccuracySelectionMethod(SelectionMethod):
    name = "training-domain validation set"

    @classmethod
    def _step_acc(self, record):
        test_env = record['args']['test_envs'][0]
        val_env_keys = []
        for i in itertools.count():
            if f'env{i}_out_acc' not in record:
                break
            if i != test_env:
                val_env_keys.append(f'env{i}_out_acc')
        test_in_acc_key = 'env{}_in_acc'.format(test_env)
        return {
            'val_acc': np.mean([record[key] for key in val_env_keys]),
            'test_acc': record[test_in_acc_key]
        }

    @classmethod
    def run_acc(self, run_records):
        test_records = get_test_records(run_records)
        if not len(test_records):
            return None
        return test_records.map(self._step_acc).argmax('val_acc')

class LeaveOneOutSelectionMethod(SelectionMethod):
    name = "leave-one-domain-out cross-validation"

    @classmethod
    def _step_acc(self, records):

        test_records = get_test_records(records)
        if len(test_records) != 1:
            return None

        test_env = test_records[0]['args']['test_envs'][0]
        n_envs = 0
        for i in itertools.count():
            if f'env{i}_out_acc' not in records[0]:
                break
            n_envs += 1
        val_accs = np.zeros(n_envs) - 1
        for r in records.filter(lambda r: len(r['args']['test_envs']) == 2):
            val_env = (set(r['args']['test_envs']) - set([test_env])).pop()
            val_accs[val_env] = r['env{}_in_acc'.format(val_env)]
        val_accs = list(val_accs[:test_env]) + list(val_accs[test_env+1:])
        if any([v==-1 for v in val_accs]):
            return None
        val_acc = np.sum(val_accs) / (n_envs-1)
        return {
            'val_acc': val_acc,
            'test_acc': test_records[0]['env{}_in_acc'.format(test_env)]
        }

    @classmethod
    def run_acc(self, records):
        step_accs = records.group('step').map(lambda step, step_records:
            self._step_acc(step_records)
        ).filter_not_none()
        if len(step_accs):
            return step_accs.argmax('val_acc')
        else:
            return None
