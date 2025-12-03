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


num_points = [3, 9, 18]
gains = [0.6,0.7,0.8]
num_trials = np.arange(1,1001,1)
counter_1 = 0
for i in np.arange(len(num_points)):
    points = num_points[i]
    for gain in gains:
        counter_2 = 0
        n_covers=3
        triangle_distance_max = []
        triangle_distance_min = []
        nontriangle_distance_max = []
        nontriangle_distance_min = []
        triangle_distance = []
        nontriangle_distance = []
        triangle_frac = []
        prob = 1-((4*(1-gain)/(3-(2*gain)))**(points-2))
        for trials in num_trials:
            counter_2 += 1
            triangle_counter = 0
            nontriangle_counter = 0
            for i in range(trials):
                data_x = np.random.uniform(0,0, int(points))
                data_y = np.random.uniform(0,1, int(points))
                data = np.vstack((data_x, data_y)).T
            

                filter_func =  mp.Projection(columns=[1])
                cover = mp.CubicalCover(n_intervals=n_covers, overlap_frac=gain) # Define cover
                clusterer = DBSCAN(min_samples=1, eps=1) # Define clusterer

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

                if len(graph.list_triangles())>=1:
                    triangle_distance.append(every_point_mean.mean())
                    triangle_distance_max.append(maxs.max())
                    triangle_distance_min.append(mins)
                    triangle_counter += 1
                elif len(graph.list_triangles())==0:
                    nontriangle_distance.append(every_point_mean.mean())
                    nontriangle_distance_max.append(maxs.max())
                    nontriangle_distance_min.append(mins)
                    nontriangle_counter += 1
        
            
            triangle_frac.append(triangle_counter/trials)
    

            counter_1 += 1
            print(counter_2)
            print(counter_1)


    #vals_to_plot = triangle_frac[9000:]
        x_vals = range(0,1000)
        print(len(x_vals))
        print(len(triangle_frac))
        plt.scatter(x_vals, triangle_frac)
        plt.axhline(y=prob, color='r', linestyle='-', label = 'P(E) = ' + str(prob)) 
        plt.xlabel('Iteration')
        plt.ylabel('Proportion of graphs that are triangles')
        #plt.title('fraction of triangle graphs for '+ str(points) + ' data points, gain = ' + str(gain))
        plt.legend()
        plt.ylim(top = 1, bottom = 0) 
        plt.savefig('Triangle graph fraction ' + str(points) + ' points, gain ' + str(gain) + ' 1k trials.pdf', dpi=300)
        plt.clf()

        column_name = str(points) + ' probability convergence, ' + str(gain) + 'gain'
        prob_data[column_name] = triangle_frac
        print(counter_2)

prob_data.to_csv('probability_convergence_1k.csv', index=False)
    


    

 


                