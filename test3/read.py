import csv
def fLoadDataMatrix(filename):
    with open(filename, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file,delimiter=',')
        data = [row for row in reader]
        i=1       
        Class=[]
        Feature=[] 
        Sample=[]
        Matrix=[]        
        for row in data:
            if(i==1):
                Feature.append(row[0:2])
                Class.append(row[2])
            else:
                Sample.append(row[0])
                Matrix.append([float(row[1]),float(row[2])])
                Class.append(row[3])
            i=i+1
        #return data
        return (Sample,Class,Feature,Matrix)


(Sample,Class,Feature,Matrix)=fLoadDataMatrix("data.csv")
#data=fLoadDataMatrix("data.csv")
print("Sample:"+str(Sample))
print("Class:"+str(Class))
print("Feature:"+str(Feature))
print("Matrix:"+str(Matrix))
       