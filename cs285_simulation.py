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

    ws=nx.newman_watts_strogatz_graph(len(voters),len(voters)/10,0.1)

    G = nx.DiGraph(ws)
    for i in range(len(voters)):
        G.node[i]['score']=voters[i]['score']
        G.node[i]['error']=voters[i]['error']

    for (a, b) in G.edges():
        
        G.edge[a][b]['weight'] = generate_rating(voters[a],voters[b])

    nx.draw(G)
    nx.draw_networkx_edge_labels(G,pos=nx.spring_layout(G))
    plt.show()

    return G

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


if __name__ == '__main__':
    main()