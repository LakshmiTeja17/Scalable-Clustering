import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn import metrics, neighbors
import time
import random
import math
import pandas as pd

def extract_boundary_points(data, k, alpha_noise, alpha_bound):
		data_len = data.shape[0]
		scores = np.full( data_len, 0.0)
		is_noise = np.full(data_len, False, dtype=bool)
		nv = np.full((data_len, data[0].size), 0.0)
		n = NearestNeighbors(n_neighbors= int(k), n_jobs=-1).fit(data)
		distances, nbrs = n.kneighbors()
		distances_av = np.average(distances, axis=-1)
		args = np.argsort(-distances_av)
		is_noise[args[:int(alpha_noise)]] = True

		for i in range(data_len):
			if is_noise[i] == True:
				continue
			nbr = nbrs[i][is_noise[nbrs[i]] == False]
			distance = distances[i][is_noise[nbrs[i]] == False]
			for j in range(len(nbr)):
				nv[i] += (data[nbr[j]] - data[i]) / distance[j]
			for j in range(len(nbr)):
				vec = (data[nbr[j]] - data[i])
				cos = np.sum(nv * vec ) / ((np.linalg.norm(nv)) * (np.linalg.norm(vec)))
				scores[i] += cos

		args = np.argsort(-scores)
		return data[args[:int(alpha_bound)]], nv[args[:int(alpha_bound)]], data[is_noise == True]

def create_guassian_data(size, spacing, sigma):
	space = sigma * spacing
	mu = [(0,0), (space, space), (0,space), (space,0)]
	cov = [[[sigma, 0], [0, sigma]] , [[sigma, 0], [0, sigma]] , [[sigma, 0], [0, sigma]], [[sigma, 0], [0, sigma]] ]
	data = np.array([np.random.multivariate_normal(mu[i], cov[i], size = size) for i in range(len(mu))])
	clusters = np.array([ [(i+1)]*size for i in range(len(mu))])
	return np.concatenate(data), np.concatenate(clusters)

def generate_circular_data(centre, radius, size):
	circle_x, circle_y = centre[0], centre[1]
	data = np.empty((size, 2))
	for i in range(size):
		r = radius * math.sqrt(random.random())
		alpha = 2 * math.pi * random.random()
		x = r * math.cos(alpha) + circle_x
		y = r * math.sin(alpha) + circle_y
		data[i] = np.array([x, y])
	return data

def create_circular_data(size, spacing, radius):
	space = 2* spacing
	mu = [(0, 0), (space, space), (0, space), (space, 0)]
	data = np.array([generate_circular_data(mu[i], radius, size = size) for i in range(len(mu))])
	clusters = np.array([ [(i+1)]*size for i in range(len(mu))])
	return np.concatenate(data), np.concatenate(clusters)

def cluster_bdr_pts(bdr_pts, nv, epsilon, m_inp):
	c_id = 0
	touched = np.full(len(bdr_pts), False, dtype=bool)
	clusters = np.full(len(bdr_pts), 0)
	n = NearestNeighbors(radius= epsilon, n_jobs=-1).fit(bdr_pts)
	nbrs = n.radius_neighbors(bdr_pts, return_distance=False)

	for i in range(len(bdr_pts)):
		if touched[i] == True:
			continue
		nbr = nbrs[i][np.sum(nv[nbrs[i]] * nv[i], axis=-1) >= 0]
		if len(nbr) < m_inp:
			continue
		touched[i] = True
		c_id += 1
		clusters[nbr] = c_id
		c_queue = set()
		c_queue.update(nbr[touched[nbr] == False].tolist())
		while len(c_queue) != 0:
			elem = c_queue.pop()
			touched[elem] = True
			nbr = nbrs[elem][np.sum(nv[nbrs[elem]] * nv[elem], axis=-1) >= 0]
			if len(nbr) >= m_inp:
				clusters[nbr] = c_id
				c_queue.update(nbr[touched[nbr] == False].tolist())
	return bdr_pts[touched == True], clusters[touched == True]

def visualize_bdr_pts(data, bdr_pts, nv, noise):
	fig, ax = plt.subplots()
	ax.scatter(x=data[:, 0], y=data[:, 1], c='red', label = 'data')
	ax.scatter(x=bdr_pts[:, 0], y=bdr_pts[:, 1], c='blue', label = 'boundary points')
	ax.scatter(x=noise[:, 0], y=noise[:, 1], c='green', label='noise')
	if len(data[0] == 2):
		for i in range(len(bdr_pts)):
			pt = bdr_pts[i]
			vec = nv[i]
			ax.arrow(pt[0], pt[1], vec[0]/ 50, vec[1]/ 50)
	ax.legend(loc='upper left')
	ax.set_title("Boundary Points")
	plt.savefig('boundary.png')
	plt.show()

def visualize_partial_clustering(data, bdr_pts, clusters):
	fig, ax = plt.subplots()
	ax.scatter(x= data[:, 0], y= data[:, 1], c= 'white', s= 50, alpha = 0.5, edgecolors = 'black')
	ax.scatter(x=bdr_pts[:, 0], y=bdr_pts[:, 1], c=clusters, cmap='viridis', s=50, alpha=1)
	ax.set_title("Partially Clustered: " + str(np.max(clusters)) + " Clusters")
	plt.savefig("partially_clustered.png")
	plt.show()

def visualize_final_clustering(data, clusters):
	fig, ax = plt.subplots()
	ax.scatter(x=data[:, 0], y=data[:, 1], c=clusters, cmap='viridis', s=50, alpha=1)
	ax.set_title("Finally Clustered: " + str(np.max(clusters)) + " Clusters")
	plt.savefig("final_clustered.png")
	plt.show()

def final_clustering(data, bdr_pts, clusters):
	n = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(bdr_pts)
	nbrs = n.kneighbors(data, return_distance = False)
	final_clusters = np.array([clusters[nbr[0]] for nbr in nbrs ])
	return final_clusters

def final_visualization(sigma, cluster_size, spacing, type):
	if type == 1:
		data, true_clusters = create_guassian_data(cluster_size, spacing, sigma)
		noise_ratio = 0.1
	elif type == 2:
		data, true_clusters = create_circular_data(cluster_size, spacing, sigma)
		noise_ratio = 0.01
	data_len = data.shape[0]
	bdr_pts, nv, noise = extract_boundary_points(data, 50, noise_ratio * data_len, data_len / 5)
	visualize_bdr_pts(data, bdr_pts, nv, noise)
	bdr_pts, bdr_clusters = cluster_bdr_pts(bdr_pts, nv, spacing / 2.5, data_len / 100)
	visualize_partial_clustering(data, bdr_pts, bdr_clusters)
	final_clusters = final_clustering(data, bdr_pts, bdr_clusters)
	visualize_final_clustering(data, final_clusters)

def comparison(spacing):
	sigma = 2
	times1 = []
	times2 = []
	accuracies1 = []
	accuracies2 = []
	for cluster_size in range(50, 501, 50):
		print("Evaluating for cluster size: ", cluster_size)
		data, true_clusters = create_guassian_data(cluster_size, spacing, sigma)
		data_len = data.shape[0]
		since = time.time()
		bdr_pts, nv, noise = extract_boundary_points(data, 50, data_len / 50, data_len / 5)
		bdr_pts, bdr_clusters = cluster_bdr_pts(bdr_pts, nv, spacing / 2.5, data_len / 100)
		final_clusters1 = final_clustering(data, bdr_pts, bdr_clusters)
		times1.append(time.time() - since)
		since = time.time()
		clustering = DBSCAN(eps=spacing / 4, min_samples=int(data_len / 50),n_jobs=-1).fit(data)
		final_clusters2 = clustering.labels_
		times2.append(time.time() - since)
		if (np.sum(final_clusters2 == -1) > 0):
			clf = neighbors.KNeighborsClassifier(1)
			clf.fit(data[final_clusters2 != -1], final_clusters2[final_clusters2 != -1])
			final_clusters2[final_clusters2 == -1] = clf.predict(data[final_clusters2 == -1])

		score1 = metrics.adjusted_rand_score(true_clusters, final_clusters1)
		accuracies1.append(score1)
		score2 = metrics.adjusted_rand_score(true_clusters, final_clusters2)
		accuracies2.append(score2)

	fig, ax = plt.subplots()
	print("SCUBI-DBSCAN Times:")
	print(times1)
	print("Normal DBSCAN Times:")
	print(times2)
	ax.plot(range(50, 501, 50), times1, label="SCUBI")
	ax.plot(range(50, 501, 50), times2, label="DBSCAN")
	ax.set_title("Times_" + str(spacing) + ".png")
	ax.legend(loc='upper right')
	plt.savefig("Times_" + str(spacing) + ".png")
	plt.show()

	fig, ax = plt.subplots()
	print("SCUBI-DBSCAN Accuracies:")
	print(accuracies1)
	print("Normal DBSCAN Accuracies:")
	print(accuracies2)
	ax.plot(range(50, 501, 50), accuracies1, label="SCUBI")
	ax.plot(range(50, 501, 50), accuracies2, label="DBSCAN")
	ax.set_title("Accuracies_" + str(spacing) + ".png")
	ax.legend(loc='upper right')
	plt.savefig("Accuracies_" + str(spacing) + ".png")
	plt.show()

print("\nType 1 to visualize the clustering.\nType 2 to get the comparison plots of DBSCAN and SCUBI-DBSCAN\n")
choice = int(input())
if choice == 1:
	sigma = int(input("Input the sigma of the data: "))
	size = int(input("Input cluster-size of the data (data has 4 clusters): "))
	spacing = int(input("Input spacing of the data as a factor of sigma: "))
	type = int(input("Input 1 for guassian data, or 2 for circular data: "))
	final_visualization(sigma, size, spacing, type)

elif choice == 2:
	spacing = int(input("Input spacing of the data as a factor of sigma (sigma is equal to 2): "))
	comparison(5)