import coverage
import keras

def calc_coverage(model, test_data, file_name):
    cov = coverage.Coverage()
    cov.start()
    
    predictions = model.predict(test_data)
    cov.stop()

    f = open(file_name, 'w')
    
    return cov.report(file=f)