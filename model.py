import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

SEED = 47


class OpinionModel:
    def __init__(self, G):
        self.G = G
        self.A = nx.adj_matrix(G).todense()
        self.N = abs(self.A).sum(axis=1)
        
        self.min_convergence_time = 10
        
        self.edges_number = self.G.number_of_edges()
        self.nodes_number = self.G.number_of_nodes()
    
    def compute_homophily(self, x):
        x = np.matrix(x).reshape((-1, 1))
        
        k = self.N  # pozor matice, ne np.array
        K = k * k.T

        X = x * x.T

        kikj2m = (K / (2*self.edges_number))
        # kidij = np.multiply(k, np.eye(self.nodes_number))

        cov = (1 / (2*self.edges_number)) * np.multiply(self.A - kikj2m, X).sum()

        # r = np.multiply(self.A - kikj2m, X).sum() / np.multiply(kidij - kikj2m, X).sum()
        return cov        

    def model(self, alpha, T, x0):
        x = x0.copy().reshape((-1, 1))

        res = [x]
        time_without_change = 0
        for t in range(T-1):
            x = alpha * np.multiply(np.sign(x), np.sqrt(abs(x))) + (1 - alpha) * np.multiply(1/self.N,  np.matmul(self.A, x))
            res.append(x)


            if np.linalg.norm(res[-1] - res[-2]) == 0:
                time_without_change += 1
            else:
                time_without_change = 0
            
            if time_without_change >= self.min_convergence_time:
                break

        return np.array(res)

    def plot_results(self, res, show=True, plot_homo=True):

        fig, ax1 = plt.subplots(figsize=(15, 4.5))

        ax1.plot(np.mean(res, axis=1), label='Mean')
        ax1.plot(np.percentile(res, 5, axis=1), c='r', label='5% perc.')
        ax1.plot(np.percentile(res, 95, axis=1), c='r', label='95% perc.')
    
        ax1.set_ylim((-1, 1))
        ax1.grid(alpha=0.3)
        ax1.legend(loc=2)


        if plot_homo:
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            homo = [self.compute_homophily(x) for x in res]
            ax2.plot(homo, '--', c='g', label='homophily')
            ax2.set_ylabel('Homophily')
            ax2.legend(loc=1)

        if show:
            plt.show()

        
    def plot_graph(self, x=None, show=True):
        nodes_plus = list(np.array(self.G.nodes)[x > 0])
        nodes_minus = list(np.array(self.G.nodes)[x < 0])
        nodes_zero = list(np.array(self.G.nodes)[x == 0])
        
        plt.figure(figsize=(15, 15))

        pos = nx.spring_layout(self.G)
        if len(nodes_plus) > 0:
            nx.draw_networkx_nodes(self.G, pos=pos, node_size=100, nodelist=nodes_plus, node_color='b', label='+1 nodes')
        if len(nodes_minus) > 0:
            nx.draw_networkx_nodes(self.G, pos=pos, node_size=100, nodelist=nodes_minus, node_color='r', label='-1 nodes')
        if len(nodes_zero) > 0:
            nx.draw_networkx_nodes(self.G, pos=pos, node_size=100, nodelist=nodes_zero, node_color='k', label='0 nodes')
        nx.draw_networkx_edges(self.G, pos=pos, width=0.15)
        plt.legend()
        plt.axis('off')
        # node_color = ['r' if xi < 0 else 'b' for xi in x]
        # nx.draw(self.G, node_color=node_color, node_size=100, width=0.3)
        if show:
            plt.show()


# G = nx.erdos_renyi_graph(100, p=0.3, seed=SEED, directed=False)
# x0 = np.random.uniform(low=-1.0, high=1.0, size=100)
#
# om = OpinionModel(G=G)
#
# res = om.model(alpha=0.5, T=100, x0=x0)
#
# om.plot_results(res)
# res.shape
#
# om.plot_graph(x=res[:, -1, 0])