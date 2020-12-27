import curve_util
import pickle

LAMBDA_VALUES = [0,0.01,0.03,0.1,0.3,1.0,3.0,10.0,30.0]
TEST_SIZES = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.15,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.0]

"""
This curve creating program for NN diagnosis has not been tested (I don't want to keep my computer running for a while), but this should work.
"""
def main():
    x = input('Learning (m) or lambda (l) curve? [m/l]:').lower()
    func = None
    params = None
    if x == 'l':
        func = curve_util.construct_learning_curves
        params = LAMBDA_VALUES
        file_name = 'lambdas.pickle'
    if x == 'm':
        func = curve_util.construct_lambda_curves
        params = TEST_SIZES
        file_name = ''
    print('Computing...')
    errors = func(params) #stored as errors = {param: [train_error,test_error], ...}
    with open(file_name,'rb') as p:
        pickle.dump(errors,p)
    print('Done!')
    return 'done'

if __name__ == '__main__':
    main()