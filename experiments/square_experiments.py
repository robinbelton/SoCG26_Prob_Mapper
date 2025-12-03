import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import DBSCAN
#import gtda.mapper as mp
import numpy as np
from sklearn.cluster import KMeans
#import gtda.diagrams.distance
from sklearn.decomposition import PCA
#import igraph as ig
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
#import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import pandas as pd
import networkx as nx
from scipy.spatial import distance_matrix


prob_data = pd.DataFrame()
triangle_distance_data = pd.DataFrame()
nontriangle_distance_data = pd.DataFrame()


num_points = [10, 20, 30]
num_trials = np.arange(1,1001,1)
counter_1 = 0
for i in np.arange(len(num_points)):
    counter_2 = 0
    points = num_points[i]
    n_covers=4
    squares_distance_max = []
    squares_distance_min = []
    nonsquare_distance_max = []
    nonsquare_distance_min = []
    square_distance = []
    nonsquare_distance = []
    square_frac = []
    #prob = 1-((4/3)*((1-0.6)/(1-(2/3)*0.6)))**(points-2)
    for trials in num_trials:
        counter_2 += 1
        square_counter = 0
        nonsquare_counter = 0
        for i in range(trials):
            data_x = np.random.uniform(0,1, int(points))
            data_y = np.random.uniform(0,1, int(points))
            data = np.vstack((data_x, data_y)).T
            

            filter_func =  mp.Projection(columns=[0,1])
            cover = mp.CubicalCover(n_intervals=n_covers, overlap_frac=0.3) # Define cover
            clusterer = DBSCAN(min_samples=1, eps=0.25) # Define clusterer

            # Initialise pipeline
            pipe_projection = mp.make_mapper_pipeline(
                filter_func=filter_func,
                cover=cover,
                clusterer=clusterer
            )

            graph = pipe_projection.fit_transform(data)

            dist_matrix = distance_matrix(data, data)
            data_points = np.arange(points)
            df = pd.DataFrame(dist_matrix, index=data_points, columns=data_points)
            df_square =df.iloc[:int(points), :int(points)]
            mins_col = []
            for point in np.arange(points):
                m = df_square[point][df[point] != 0].min()
                mins_col.append(m)


            mins = min(mins_col)
            maxs = df_square.max()
            every_point_mean = df_square.mean()

            squares = graph.simple_cycles(min=4,max=4)

            if len(squares) >= 1:
                square_distance.append(every_point_mean.mean())
                squares_distance_max.append(maxs.max())
                squares_distance_min.append(mins)
                square_counter += 1
            elif len(squares)==0:
                nonsquare_distance.append(every_point_mean.mean())
                nonsquare_distance_max.append(maxs.max())
                nonsquare_distance_min.append(mins)
                nonsquare_counter += 1
        
            
        square_frac.append(square_counter /trials)
    

        counter_1 += 1
        print(counter_2)
        print(counter_1)

       

    x_vals = range(0,1000)
    print(len(x_vals))
    print(len(square_frac))
    #plt.axhline(y=prob, color='r', linestyle='-', label = str(prob)) 
    plt.xlabel('iteration')
    plt.ylabel('Proportion of Graphs that are Squares')
    #plt.title('Proportion Squares for '+ str(points) + ' data points')
    #plt.legend()
    #plt.ylim(top = prob+0.05, bottom = prob-0.05) 
    plt.savefig('Square graph fraction ' + str(points) + ' points 1k trials, 0.5 overlap.pdf', dpi=300)
    plt.clf()

    column_name = str(points) + ' probability convergence'
    prob_data[column_name] = square_frac
    print(counter_2)

prob_data.to_csv('probability_convergence_square_1k.csv', index=False)
    


    

 


                