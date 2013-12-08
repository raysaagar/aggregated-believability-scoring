"""
Simulation

1) Generate a list of people
    - hidden score
    - hidden accuracy
2) Generate rating graph
    - directed graph where each person has a random number of outgoing edges 
    - ratings should be randomly drawn from distribution (mean: hidden score, variance: inversely prop to hidden accuracy (1-a))


"""
from scipy.stats import norm
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import mdp

N = 100

def adjust(n, lower=0, upper=10):
    if n < lower:
        n = lower
    if n > upper:
       n = upper
    return n

# voters is our list of people to convert to nodes later
def generate_voters(number=100, score_mean=5., score_variance=1.5, error_mean=2., error_variance=0.8):
    scores_rv = norm(score_mean, score_variance)
    scores = scores_rv.rvs(size=number)
    scores = [adjust(i) for i in scores]
    
    errors_rv = norm(error_mean, error_variance)
    errors =  errors_rv.rvs(size=number)
    errors = [adjust(i) for i in errors]
    return [{'score':score, 'error':error} for (score, error) in zip(scores, errors)]

def generate_rating(reviewer, victim):
    rv = norm(victim["score"], reviewer["error"])
    rating = rv.rvs(size=1)[0]
    return adjust(rating)

def generate_graph(voters):
    
    # generate edges randomly
    # n, k, p, [seed]
    # n = nodes

    ws=nx.newman_watts_strogatz_graph(len(voters),len(voters)/10,1)

    G = nx.DiGraph(ws)
    for i in range(len(voters)):
        G.node[i]['score']=voters[i]['score']
        G.node[i]['error']=voters[i]['error']

    for (a, b) in G.edges():
        
        G.edge[a][b]['weight'] = generate_rating(voters[a],voters[b])

    #nx.draw(G)
    #nx.draw_networkx_edge_labels(G,pos=nx.spring_layout(G))
    #plt.show()

    return G

def generate_similarity_graph(rating_graph):
    size = len(rating_graph.nodes())
    ratings = np.zeros((size, size))
    for a, b in rating_graph.edges():
        ratings[a][b] = rating_graph.edge[a][b]['weight']
    # use 3 principal components in the PCA
    pcanode1 = mdp.nodes.PCANode(output_dim=3)
    # train the model
    pcanode1.train(ratings)
    pcanode1.stop_training()
    print 'explained variance:', pcanode1.explained_variance
    # run the model - get the results
    pca = pcanode1(ratings) 
    sim = np.zeros((size, size))
    # building the similarity scores from PCA components
    w = [0.6, 0.3, 0.1]
    for i in range(size):
        for j in range(size):
            if i == j:
                continue
            diffs = [abs(pca[i][k]-pca[j][k])*w[k] for k in range(len(w))]
            sim[i][j] = 7-sum(diffs)
    sims = [item for sublist in sim for item in sublist]
    plt.hist(sims, bins=[-10+0.5*i for i in range(40)])
    plt.show()
    # TODO - joseph wants to change thresholding to a top % model
    sim_thresh = 3.
    G = nx.DiGraph()
    G.node = rating_graph.node
    for i in range(size):
        for j in range(size):
            if i == j:
                continue
            if abs(sim[i][j]) > sim_thresh:
                G.add_edge(i, j, weight=sim[i][j])
    nx.draw(G)
    plt.show()
    return G

def normalize_edges(G):
    for a in G.nodes():
        total = sum([abs(G.edge[a][b]['weight']) for b in G[a]])
        for b in G[a]:
            G.edge[a][b]['weight'] /= total
def main():
    print "Generating Voters..."
    voters = generate_voters(N)

    # print "TEST: CREATING A NEW RATING"
    # print 'rater:', voters[0]
    # print 'ratee:', voters[1]
    # rating = generate_rating(voters[0], voters[1])
    # print 'rating:', rating
    # print 'error=', abs(voters[1]['score']-rating)

    print "Generating Graph..."
    G = generate_graph(voters)
    H = generate_similarity_graph(G)
    normalize_edges(H)

if __name__ == '__main__':
    main()