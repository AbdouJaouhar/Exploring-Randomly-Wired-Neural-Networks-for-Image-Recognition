class WattsStrogatz(object):
    def __init__(self, N,K,P):
        self.N = N
        self.K = K
        self.P = P
        self.nodes = {i : Node(i) for i in range(N)}
        self.edges = {i : [(i-k)%N for k in range(-K//2, K//2+1) if k !=0] for i in range(N)}
        
        self.ClockwiseRewiring()
        self.make_acyclic()
    
    def __len__(self):
        return self.N

    def ClockwiseRewiring(self):
        for i, v in self.nodes.items():
            for j in range(i+1,i+self.K//2+1):
                if j in self.edges[i]:
                    is_rewired = self.P > np.random.rand()
                    if is_rewired:
                        new_node = np.random.choice([k for k in range(self.N) if k != i and k not in self.edges[i]])
                        self.edges[i][self.edges[i].index(j)] = new_node

    def get_inputs_node(self, node):
        inputs_node = []

        for node_i, node_edges in self.edges.items():
            if node in node_edges and node_i not in inputs_node:
                inputs_node.append(node_i)

        return inputs_node

    def is_cyclic(self):
        visited = [False for i in range(self.N)]
        breadCrumb = [False for i in range(self.N)]
        
        def submodule(root, m_visited, m_breadCrumb):
            if breadCrumb[root] == True:
                return True
            
            if visited[root] == True:
                return False
            
            m_visited[root] = True
            m_breadCrumb[root] = True
            
            if len(self.edges[root]) != 0:
                for child in self.edges[root]:
                    if submodule(child, m_visited, m_breadCrumb):
                        return True
            m_breadCrumb[root] = False
            return False
        return submodule(0, visited, breadCrumb)
    
    def make_acyclic(self):
        for node, n_edge in self.edges.items():
            for n in n_edge:
                if node > n:
                    if node not in self.edges[n]:
                        self.edges[n].append(node)
                    temp = []
                    for u in self.edges[node]:
                        if u != n:
                            temp.append(u)
                    self.edges[node] = temp
    
    def topological_sort(self):
        visited = {i : False for i in range(self.N)}
        stack = []

        def topological_sort_rec(i,visited,stack): 
            visited[i] = True
      
            for k in self.edges[i]: 
                if visited[i] == False: 
                    topological_sort_rec(k,visited,stack) 
      
            stack.insert(0,i) 


        for i in range(self.N):
            if visited[i] == False:
                topological_sort_rec(i, visited, stack)
        
        return stack