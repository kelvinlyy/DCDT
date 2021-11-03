import coverage
import keras
import pandas as pd

class Class_Function:    
    
    def __init__(self, name, lines):
        self.name = name
        self.lines = lines
        self.length = len(lines)
        self.coverage = self.calc_coverage()
        
    def calc_coverage(self):
        if (self.length == 0):
            return 0
        else:
            c = 0
            for line in self.lines:
                if (line[1] == '1'):
                    c += 1
            return c/self.length

class Package:
    
    def __init__(self, name):
        self.name = name
        self.classes = []
        self.coverage = 0
    
    def calc_coverage(self):
        tmp = 0
        count = 0
        for _ in self.classes:
            tmp += _.coverage
            count += 1
        return tmp/count

class Coverage_Table: 
    
    def __init__(self):
        self.packages = []
        self.coverage = 0
        
    def calc_coverage(self):
        tmp = 0
        count = 0
        for _ in self.packages:
            tmp += _.coverage
            count += 1
        return tmp/count
        
    def print_details(self):
        for i in self.packages:
            print(i.name)
            for j in i.classes:
                print(f'\t{j.name}')
                for k in j.lines:
                    print(f'\t\t{k}')
        
def init_CovTable():
    
    import xml.etree.ElementTree as ET
    tree = ET.parse('coverage.xml')
    packages = tree.getroot()[1]

    cov_table = Coverage_Table()
    for package in packages:
        new_package = Package(package.get('name'))
        for class_function in package[0]:
            line_num = []
            line_hit = []
            for line in class_function[1]:
                line_num.append(line.get('number'))
                line_hit.append(line.get('hits'))
            new_class = Class_Function(class_function.get('filename'), pd.DataFrame([line_num, line_hit]).T.values.tolist())
            new_package.classes.append(new_class)
        new_package.coverage = new_package.calc_coverage()
        cov_table.packages.append(new_package)
    cov_table.coverage = cov_table.calc_coverage()
    
    return cov_table
    
def calc_coverage(model, test_data, file_name):
    cov = coverage.Coverage()
    cov.start()
    
    predictions = model.predict(test_data)
    cov.stop()
    
    return cov.xml_report(outfile = file_name)