import numpy as np
import argparse
import os
import liblinear.liblinearutil as liblinearsvm


def main():
    parser = argparse.ArgumentParser('svm_perf')
    parser.add_argument('--output-dir', type=str, default='log/HMDB51_Linear')
    parser.add_argument('--num_replica', type=int, default=8)
    parser.add_argument('--cost', type=float, default=100)
    parser.add_argument('--primal', action='store_true', default=True)
    args = parser.parse_args()


    feat_train=np.load(os.path.join(args.output_dir, 'feature_train.npy'))
    feat_train_cls=np.load(os.path.join(args.output_dir, 'feature_train_cls.npy'))

    feat_val=np.load(os.path.join(args.output_dir, 'feature_val.npy'))
    feat_val_cls=np.load(os.path.join(args.output_dir, 'feature_val_cls.npy'))

    print('feat_val: {}'.format(feat_val.shape))
    print('feat_val_cls: {}'.format(feat_val_cls.shape))

    print('form svm problem')
    svm_problem = liblinearsvm.problem(feat_train_cls, feat_train)
    if args.primal:
        print('L2-regularized L2-loss support vector classification (primal), cost={}'.format(args.cost))
        svm_parameter = liblinearsvm.parameter('-s 2 -n 32 -c {}'.format(args.cost))
        svm_filename = 'multicore_linearsvm_primal_c{}.svmmodel'.format(args.cost)
    else:
        print('L2-regularized L2-loss support vector classification (dual), cost={}'.format(args.cost))
        svm_parameter = liblinearsvm.parameter('-s 1 -n 32 -c {}'.format(args.cost))
        svm_filename = 'multicore_linearsvm_dual_c{}.svmmodel'.format(args.cost)
    print('train svm')
    svm_model = liblinearsvm.train(svm_problem, svm_parameter)
    print('save svm')
    liblinearsvm.save_model(os.path.join(args.output_dir, svm_filename), svm_model)
    print('eval svm')
    pd_label, pd_acc, pd_val = liblinearsvm.predict(feat_val_cls, feat_val, svm_model)
    eval_acc, eval_mse, eval_scc = liblinearsvm.evaluations(feat_val_cls, pd_label)
    print('{}/{}'.format(pd_acc, eval_acc))
    with open(os.path.join(args.output_dir, svm_filename + '.txt'), 'w') as f:
        f.write('{}/{}'.format(pd_acc, eval_acc))
    print('Done')


if __name__ == '__main__':
    main()