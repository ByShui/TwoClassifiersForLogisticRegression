from numpy import *

def TxtToNumpy(filename):
    file = open(filename)
    file_lines_list = file.readlines()
    number_of_file_lines = len(file_lines_list)
    dataMat = zeros((number_of_file_lines, 3))
    labelList = []
    index = 0
    for line in file_lines_list:
        line = line.strip()
        line_list = line.split('\t')
        dataMat[index, :] = line_list[0:3]
        labelList.append(int(line_list[-1]))
        index += 1
    return dataMat, labelList

if __name__ == "__main__":
    print("Code Run As A Program")
