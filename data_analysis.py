import csv
import numpy as np
import matplotlib.pyplot as plt


wine_features = {
	0:"fixed acidity",
	1:"volatile acidity",
	2:"citric acid",
	3:"residual sugar",
	4:"chlorides",
	5:"free sulfur dioxide",
	6:"total sulfur dioxide",
    7:"density",
    8:"pH",
    9:"sulphates",
    10:"alcohol",
    11:"quality"
    
}

def import_wine_data(ifname):
    with open(ifname, 'r') as ifile:
        datareader = csv.reader(ifile, delimiter=';')
        # we want to avoid importing the header line.
        # instead we'll print it to screen
        header = next(datareader)
        #print("Importing data with fields:\n\t" + ",".join(header))
        # create an empty list to store each row of data
        data = []
        for row in datareader:
            #print("row = %r" % (row,))
            # for each row of data 
            # convert each element (from string) to float type
            row_of_floats = list(map(float,row))
            #print("row_of_floats = %r" % (row_of_floats,))
            # now store in our data list
            data.append(row_of_floats)
        #print("There are %d entries" % len(data))
        # convert the data (list object) into a numpy array.
        data_as_array = np.array(data)
        # return this array to caller
        return data_as_array

def main(ifname, row):
    row = int(row)
    data = import_wine_data(ifname)
    y = data[:,row-1]
    x = data[:,11]
    plt.plot(x,y,'o')
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
    plt.ylabel(wine_features[row-1])
    plt.xlabel('Quality')
    plt.title('Influence of '+wine_features[row-1]+' on the quality')
    plt.show()
            
    #histogram_durations(data)
    #scatter_duration_versus_wait(data)
    #plot_time_evolution(data)
    #scatter_adjacent_datapoints(data)
    #plt.show()

	

if __name__ == '__main__':
    import sys
    # this allows you to pass the file name as the first argument when you call
    # your script from the command line
    # so to run this script use:(
    # python old_faithful.py old_faithful.tsv
    main(sys.argv[1],sys.argv[2])
    
