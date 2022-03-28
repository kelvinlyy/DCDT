import coverage
from tensorflow import keras
import pandas as pd

def lines2Vector(lines):
    lineVector = []
    if lines:
        n = 1
        m = 0
        while (n <= lines[-1]):
            if lines[m] == n:
                lineVector.append(1)
                m += 1
            else:
                lineVector.append(0)
            n += 1
    return lineVector

def calc_coverage(model, test_data):
    cov = coverage.Coverage()
    cov.start()
    
    predictions = model.predict(test_data)
    cov.stop()    
    covData = cov.get_data()
    covList = [(files, sorted(covData.lines(files))) for files in covData.measured_files()]      
    
    return covList
 

def total_lines(covList):
    cnt = 0
    for _ in covList:
        cnt += len(_[1])
    return cnt

def delta_lines(covList1, covList2):
    cnt = 0
    for files in range(len(covList1)):
        a = set(covList1[files][1])
        b = set(covList2[files][1])
        cnt += len(a.difference(b))
        cnt += len(b.difference(a))    
        
    return cnt

# class Class_Function:    
    
#     def __init__(self, name, lines):
#         self.name = name
#         self.lines = lines
#         self.length = len(lines)
#         self.coverage = self.calc_coverage()
        
#     def calc_coverage(self):
#         if (self.length == 0):
#             return 0
#         else:
#             c = 0
#             for line in self.lines:
#                 if (line[1] == '1'):
#                     c += 1
#             return c/self.length

# class Package:
    
#     def __init__(self, name):
#         self.name = name
#         self.classes = []
#         self.coverage = 0
    
#     def calc_coverage(self):
#         tmp = 0
#         count = 0
#         for _ in self.classes:
#             tmp += _.coverage
#             count += 1
#         return tmp/count

# class Coverage_Table: 
    
#     def __init__(self):
#         self.packages = []
#         self.coverage = 0
        
#     def calc_coverage(self):
#         tmp = 0
#         count = 0
#         for _ in self.packages:
#             tmp += _.coverage
#             count += 1
#         return tmp/count
        
#     def print_details(self):
#         for i in self.packages:
#             print(i.name)
#             for j in i.classes:
#                 print(f'\t{j.name}')
#                 for k in j.lines:
#                     print(f'\t\t{k}')
        
# def init_CovTable():
    
#     import xml.etree.ElementTree as ET
#     tree = ET.parse('coverage.xml')
#     packages = tree.getroot()[1]

#     cov_table = Coverage_Table()
#     for package in packages:
#         new_package = Package(package.get('name'))
#         for class_function in package[0]:
#             line_num = []
#             line_hit = []
#             for line in class_function[1]:
#                 line_num.append(line.get('number'))
#                 line_hit.append(line.get('hits'))
#             new_class = Class_Function(class_function.get('filename'), pd.DataFrame([line_num, line_hit]).T.values.tolist())
#             new_package.classes.append(new_class)
#         new_package.coverage = new_package.calc_coverage()
#         cov_table.packages.append(new_package)
#     cov_table.coverage = cov_table.calc_coverage()
    
#     return cov_table