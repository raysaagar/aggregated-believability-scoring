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
from scipy.stats import pearsonr
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
    w = [0.6, 0.3, 0.1] # WEIGHTS 
    for i in range(size):
        for j in range(size):
            if i == j:
                continue
            diffs = [abs(pca[i][k]-pca[j][k])*w[k] for k in range(len(w))]
            sim[i][j] = 7-sum(diffs) # 7 is magic centralizing factor - wat hax doge
    sims = [item for sublist in sim for item in sublist]
    # plt.hist(sims, bins=[-10+0.5*i for i in range(40)])
    # plt.show()
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
    # nx.draw(G)
    # plt.show()
    return G

def normalize_edges(G):
    for a in G.nodes():
        total = sum([abs(G.edge[a][b]['weight']) for b in G[a]])
        for b in G[a]:
            G.edge[a][b]['weight'] /= total

# step 1a - assign each node credibility of 1/n
# step 1b - create a copy of G as G'
# step 2 - each node's cred in G' is cred = max(sum(edge weight * source cred in G FORALL incoming edges),0)
# step 3 - G = G'
# step 4 - if convergence then GOTO 7
# step 5 - for each node, cred = (1-p) * cred 
# step 5b - for each node with cred > 0, cred = cred + p/m, where m is set of all nodes where cred != 0 (this is the random walker)
# step 6 - GOTO 2 
# step 7 - return all trust valus and go home
def assign_credibility(graph):
    
    G = graph
    size = len(G.nodes())
    
    threshold = 1./(size*100)
    # threshold = 0.001
    mixing_factor = 0.01
    converged = False
    step_counter = 0

    old_changes = []
    old_cred_list = {}

    # step 1
    for a in G.nodes():
        G.node[a]['credibility'] = 1./(size)

    while True:
        max_change = 0

        converged = True
        # step 1b
        H = G.copy()

        nonzero_counter = 0
        # step 2
        for node in G.nodes():
            old_cred_list[node] = G.node[node]['credibility']
            old_cred = G.node[node]['credibility']
            new_cred = 0.
            for src in G.predecessors(node):
                new_cred += G.node[src]['credibility'] * G.edge[src][node]['weight']
            new_cred = max(new_cred, 0)
            H.node[node]['credibility'] = new_cred

            # track nonzero cred nodes for step 5
            if new_cred > 0:
                nonzero_counter += 1

        # make initial copy here with the smoothed results
        G = H.copy()

        # step 4

        # step 5 - propogate a random walk
        for node in H.nodes():

            if H.node[node]['credibility'] > 0:
                # step 5a
                H.node[node]['credibility'] *= (1. - mixing_factor)
                # step 5b
                H.node[node]['credibility'] += mixing_factor/nonzero_counter

            # check for convergence here
            if abs(H.node[node]['credibility'] - old_cred_list[node]) > threshold:
                max_change = max(max_change, abs(new_cred - old_cred))
                converged = False
        # print "Step " + str(step_counter) + " Max Change " + str(max_change)
        step_counter += 1

        # if we converged, don't keep the random walk changes
        if converged:
            # for node in H.nodes():
            #     print H.node[node]['credibility']
            return H
            break

        G = H.copy() # TODO: do we need .copy?

        # step 6 is end while

    # step 7
    # return G


def calculate_correlation(rating_graph, cred_graph):
    cred_list = []
    error_list = []
    
    for node in cred_graph.nodes():
        cred_list.append(cred_graph.node[node]['credibility'])
        error_list.append(rating_graph.node[node]['error'])

    correlation = pearsonr(error_list,cred_list)
    # print cred_list
    # plt.scatter(error_list,cred_list)
    # plt.xlim(-2, 2)
    #plt.ylim(0.0095, 0.0105)
    # plt.show()
    return correlation[0]

def main():
    correlations = []
    ntrials = 100
    for i in range(ntrials):
        print i
        # print "Generating Voters..."
        voters = generate_voters(N)

        # print "Generating Graphs..."
        G = generate_graph(voters)
        rating_graph = G.copy()
        
        H = generate_similarity_graph(G)

        # print "Normalizing Edges..."
        normalize_edges(H)

        # print "Calculating credibility scores"
        cred_graph = assign_credibility(H)

        correlations.append(calculate_correlation(rating_graph,cred_graph))

    plt.hist(correlations, bins=[-1+0.05*i for i in range(40)])

    plt.show()
    print "DONE"

if __name__ == '__main__':
    main()