import numpy
import random
import math
from collections import defaultdict
from neo4j import GraphDatabase, basic_auth
class Graph: 
   
    def __init__(self,graph): 
        self.graph = graph # residual graph 
        self. ROW = len(graph) 
          
   
    '''Returns true if there is a path from source 's' to sink 't' in 
    residual graph. Also fills parent[] to store the path '''
    def BFS(self,s, t, parent): 
  
        visited =[False]*(self.ROW) 
        queue=[] 
        queue.append(s) 
        visited[s] = True
    
        while queue: 
  
            u = queue.pop(0) 
          
            for ind, val in enumerate(self.graph[u]): 
                if visited[ind] == False and val > 0 : 
                    queue.append(ind) 
                    visited[ind] = True
                    parent[ind] = u 
  
        return True if visited[t] else False
              
      
    # Returns tne maximum flow from s to t in the given graph 
    def FordFulkerson(self, source, sink): 
  
        parent = [-1]*(self.ROW) 
  
        max_flow = 0
  
        while self.BFS(source, sink, parent) : 
  
            path_flow = float("Inf") 
            s = sink 
            while(s !=  source): 
                path_flow = min (path_flow, self.graph[parent[s]][s]) 
                s = parent[s] 
  
            # Add path flow to overall flow 
            max_flow +=  path_flow 
  
            # update residual capacities of the edges
            v = sink 
            while(v !=  source): 
                u = parent[v] 
                self.graph[u][v] -= path_flow 
                self.graph[v][u] += path_flow 
                v = parent[v] 
  
        return max_flow
def haversine(latx,laty,longx,longy):
    a = (pow(math.sin((laty-latx) / 2), 2) + 
         pow(math.sin((longy-longx) / 2), 2) * 
             math.cos(latx) * math.cos(laty))
    rad=6371
    c=2*math.asin(math.sqrt(a))
    return rad*c
#polynomial generation.

deg = random.uniform(0.45,0.5)
print(deg)

#s

xs = [1.1,2.6,4,5]
ys = [2,4,5,6]

zs = numpy.polyfit(xs,ys,deg)
print(zs)

#p

xp = [1.1,2.6,4,5]
yp = [1,2.5,4,5]

zp = numpy.polyfit(xp,yp,deg)
print(zp)

#input s & p

s = [7,10,11]
p = [2,5,9]

t = numpy.subtract(s,p)
print(t)

inner = t/abs(zp-zs) 
ideg = 1/deg

x = pow(inner,ideg)
print(x)

# Generating random amplitudes to calculates magnitudes

amp = []
for i in range(3):
    ramp = random.uniform(5,6)
    amp.append(ramp)

print("\nAmplitudes are")
print(amp,"\n")

# Calculating the magnitudes

mag = []

for i in range(3):
    magc = (math.log(amp[i],10)) + (2.56)*(math.log(x[i],10)) - 1.67
    mag.append(magc)

print("Magnitudes are")
print(mag)

# Finding the max and min magnitudes and corresponding indexes

max1 = 0
for i in range(3):
    if(mag[i]>max1):
        max1 = mag[i]
for i in range(3):
    if(mag[i] == max1):
        maxindex = i

min1 = 1000
for i in range(3):
    if(mag[i]<min1):
        min1 = mag[i]
for i in range(3):
    if(mag[i] == min1):
        minindex = i

# Finding the range of the earthquake effect

maxdistance = x[maxindex]
mindistance = x[minindex]

print("\nThe range is")
erange = maxdistance - mindistance
print(erange)

# Finding the probability where the earthquake can occur

rx = [round(i) for i in x]
rx.sort()

countlist = []

for i in rx:
    count = 0
    for j in rx:
        if(i == j):
            count+=1

    countlist.append(count)

maxfreq = 0
for i in countlist:
    if(i > maxfreq):
        maxfreq = i

for i in countlist:
    if(i == maxfreq):
        maxfind = i

print("\nThe probable place of happening of earthquake is")
print(rx[maxfind]) 
   
# Create a graph given in the above diagram 
driver=GraphDatabase.driver("bolt://localhost:7687",auth=basic_auth("neo4j","flashwareayan12345"))
session=driver.session()
result=list(session.run("MATCH ((n)-[r]->(m)) RETURN n.name,n.lat,n.long,n.pop,r.weight,m.name,m.lat,m.long,m.pop"))
L=[]
L1=[]
L2=[]
for record in result:
    K=[]
    K.extend([record["n.name"],record["n.lat"],record["n.long"],record["n.pop"],record["r.weight"],record["m.name"],record["m.lat"],record["m.long"],record["m.pop"]])
    L.append(K)
    L2.append(record["n.name"])
L1=list(set(tuple(sub) for sub in L))
print(L1)
L4=list(dict.fromkeys(L2))
print(L4)
print()
x=len(L4)
adj=[]
for i in range(x):
    row=[]
    for k in range(x):
        row.append(0)
    adj.append(row)
for path in L:
    adj[path[0]][path[5]]=path[4]
#printing adjacency matrix
for i in range(x):
    for k in range(x):
        print(adj[i][k],"\t",end="")
    print()
#finding the epicentre
y=random.randrange(0,x-1)
latx=0.0
longx=0.0
for i in L:
    if i[0]==y:
        latx=i[1]
        longx=i[2]
distmin=99999
epi=y
for i in L4:
    if i != y:
        for k in L:
            if k[0]==i:
                laty=k[1]
                longy=k[2]
                d1=round(haversine(latx,laty,longx,longy))
                if d1<=rx[maxfind]:
                    distmin=d1
                    epi=i
                    latx=k[1]
                    longx=k[2]
                    
#storing sources and destinations
s1=[]
s2=[]
for i in L:
    dc=round(haversine(latx,i[1],longx,i[2]))
    if dc>erange:
             s2.append(i[0])
s3=list(dict.fromkeys(s2))
for i in L4:
    if i not in s3:
        s1.append(i)
#calculating max flow
for i1 in s1:
    smalldist=99999
    lat1=0.0
    lat2=0.0
    long1=0.0
    long2=0.0
    for i2 in s3:
        for k1 in L:
            if k1[0]==i1:
                lat1=k1[1]
                long1=k1[2]
            if k1[0]==i2:
                lat2=k1[1]
                long2=k1[2]
        d2=round(haversine(lat1,lat2,long1,long2))
        if(d2<smalldist):
            smalldist=d2
            x2=i2
    lx=adj.copy()
    pop1=0
    for k2 in L:
        if i1==k2[0]:
            pop1=k2[3]
    g=Graph(lx)
    r=g.FordFulkerson(i1,x2)
    if(r>pop1):
        f1=1
    else:
        f1=math.ceil(pop1/r)
    print("The max flow for the node",i1,"=", r)
    print("Total no. of flows=",f1)
    
