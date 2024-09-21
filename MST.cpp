//
// Created by agalex on 7/20/23.
//

#include <cstdio>
#include <cstdbool>
#include <climits>
#include <cstdlib>
#include <cmath>
#include <queue>
#include <set>
#include <stack>
#include <iostream>
#include <unordered_map>
#include<bits/stdc++.h>
using namespace std;

// Structure to represent a node in the adjacency list
/*typedef struct Node
{
    int dest;
    double weight;
    struct Node* next;
} Node;

// Structure to represent an adjacency list
typedef struct List
{
    Node *head;
} List;

// Structure to represent a graph. A graph is an array of adjacency lists
typedef struct Graph
{
    int V;
    List* array;
} Graph;

Node* newAdjListNode(int dest, int weight)
{
    Node* newNode = (Node*) malloc(sizeof(Node));
    newNode->dest = dest;
    newNode->weight = weight;
    newNode->next = NULL;
    return newNode;
}

Graph* createGraph(int V)
{
    Graph* graph = (Graph*) malloc(sizeof(Graph));
    graph->V = V;
    graph->array = (List*) malloc(V * sizeof(List));

    int i;
    for (i = 0; i < V; ++i)
        graph->array[i].head = NULL;

    return graph;
}

void addEdge(Graph* graph, int src, int dest, int weight)
{
    Node* newNode = newAdjListNode(dest, weight);
    newNode->next = graph->array[src].head;
    graph->array[src].head = newNode;

    newNode = newAdjListNode(src, weight);
    newNode->next = graph->array[dest].head;
    graph->array[dest].head = newNode;
}

void MSTOrientPointCloud(float* nx, float* ny, float* nz, Graph* graph){
    struct GeoElement {
        double geodistance;
        int index;
    };

    auto cmp = [](const GeoElement left, const GeoElement right) { return left.geodistance > right.geodistance; };
    std::priority_queue<GeoElement, std::vector<GeoElement>, decltype(cmp)> VLIST(cmp);

    int V = graph->V;
    int* parent = new int[V];
    double* key = new double[V];
    bool* mstSet = new bool[V];

    for (int i = 0; i < V; i++)
        key[i] = 1e10, mstSet[i] = false;

    key[0] = 0;
    parent[0] = -1;

    for (int i = 0; i < 1; i++)
    {
        GeoElement gi{1e10, i};
        VLIST.push(gi);
    }

    while (!VLIST.empty())
    {
        GeoElement ge = VLIST.top();
        VLIST.pop();
        int u = ge.index;
        mstSet[u] = true;

        Node* node = graph->array[u].head;
        while(node != NULL)
        {
            int v = node->dest;
            if (mstSet[v] == false && node->weight < key[v]) {
                parent[v] = u;
                key[v] = node->weight;
                GeoElement gi{node->weight, v};
                VLIST.push(gi);
                if (nx[u] * nx[v] + ny[u] * ny[v] + nz[u] * nz[v] < 0){
                    nx[v] = -nx[v];
                    ny[v] = -ny[v];
                    nz[v] = -nz[v];
                }
            }
            node = node->next;
        }
    }
}*/

double norm(int i, int j, float* x, float* y, float* z){
    return (((x[i] - x[j]) * (x[i] - x[j]) + (y[i] - y[j]) * (y[i] - y[j]) + (z[i] - z[j]) * (z[i] - z[j])));
}

void MSTOrientPointCloud(int* knn, float* nx, float* ny, float* nz, float* x, float* y, float* z, int V, int k, int k_use){
    struct GeoElement {
        double geodistance;
        int index;
    };

    auto cmp = [](const GeoElement left, const GeoElement right) { return left.geodistance > right.geodistance; };
    std::priority_queue<GeoElement, std::vector<GeoElement>, decltype(cmp)> VLIST(cmp);


    int* parent = new int[V];
    double* key = new double[V];
    bool* mstSet = new bool[V];

    for (int i = 0; i < V; i++)
        key[i] = 1e10, mstSet[i] = false;

    key[0] = 0;
    parent[0] = -1;

    for (int i = 0; i < 1; i++)
    {
        GeoElement gi{1e10, i};
        VLIST.push(gi);
    }

    while (!VLIST.empty())
    {
        GeoElement ge = VLIST.top();
        VLIST.pop();
        int u = ge.index;
        mstSet[u] = true;
        for (int j = 0; j < k_use; j++)
        {
            int v = knn[u * k + j];
            if (v >= 0) {
                double weight = norm(u, v, x, y, z);
                if (mstSet[v] == false && weight < key[v]) {
                    parent[v] = u;
                    key[v] = weight;
                    GeoElement gi{weight, v};
                    VLIST.push(gi);
                }
            }
        }
    }
}

std::pair<int, int> order(int i, int j){
    if (i < j) return {i, j};
    else return {j, i};
}

// C++ program for Kruskal's algorithm to find Minimum
// Spanning Tree of a given connected, undirected and
// weighted graph

// To represent Disjoint Sets
struct DisjointSets
{
    int *parent, *rnk;
    int n;

    // Constructor.
    DisjointSets(int n)
    {
        // Allocate memory
        this->n = n;
        parent = new int[n+1];
        rnk = new int[n+1];

        // Initially, all vertices are in
        // different sets and have rank 0.
        for (int i = 0; i <= n; i++)
        {
            rnk[i] = 0;

            //every element is parent of itself
            parent[i] = i;
        }
    }

    // Find the parent of a node 'u'
    // Path Compression
    int find(int u)
    {
        /* Make the parent of the nodes in the path
        from u--> parent[u] point to parent[u] */
        if (u != parent[u])
            parent[u] = find(parent[u]);
        return parent[u];
    }

    // Union by rank
    void merge(int x, int y)
    {
        x = find(x), y = find(y);

        /* Make tree with smaller height
        a subtree of the other tree */
        if (rnk[x] > rnk[y])
            parent[y] = x;
        else // If rnk[x] <= rnk[y]
            parent[x] = y;

        if (rnk[x] == rnk[y])
            rnk[y]++;
    }
};

// Creating shortcut for an integer pair
typedef pair<int, int> iPair;

// Structure to represent a graph
struct Graph
{
	int V, E;
	vector< pair<double, iPair> > edges;

	// Constructor
	Graph(int V)
	{
		this->V = V;
        E = 0;
	}

	// Utility function to add an edge
	void addEdge(int u, int v, double w)
	{
		edges.push_back({w, {u, v}});
        E++;
	}

	// Function to find MST using Kruskal's
	// MST algorithm
	double kruskalMST(float* nx, float* ny, float* nz, int* knn, int k, int k_use, float* x, float* y, float *z);
    void reverseOrientation(int root, DisjointSets& ds, float* nx, float* ny, float* nz){
        for (int i = 0; i < V; i++){
            if (ds.find(i) == root){
                nx[i] = -nx[i];
                ny[i] = -ny[i];
                nz[i] = -nz[i];
            }
        }
    }

    void orientSpanningTree(int root , float* nx, float* ny, float* nz, std::vector<std::vector<int>>& mst){
        std::stack<int> st;
        std::vector<bool> processed(V, false);
        st.push(root);
        while (!st.empty()){
            int u = st.top();
            st.pop();
            for (auto v : mst[u]){
                if (!processed[v]) {
                    if (nx[u] * nx[v] + ny[u] * ny[v] + nz[u] * nz[v] < 0) {
                        nx[v] = -nx[v];
                        ny[v] = -ny[v];
                        nz[v] = -nz[v];
                    }
                    st.push(v);
                }
            }
            processed[u] = true;
        }
    }
};

/* Functions returns weight of the MST*/

double Graph::kruskalMST(float* nx, float* ny, float* nz, int* knn, int k, int k_use, float* x, float* y, float *z)
{
	std::vector<std::vector<int>> mst(V);
    double mst_wt = 0.0; // Initialize result

	// Sort edges in increasing order on basis of cost
	sort(edges.begin(), edges.end());

	// Create disjoint sets
	DisjointSets ds(V);

	// Iterate through all sorted edges
	vector< pair<double, iPair> >::iterator it;
	for (it=edges.begin(); it!=edges.end(); it++)
	{
		int u = it->second.first;
		int v = it->second.second;

		int set_u = ds.find(u);
		int set_v = ds.find(v);

		// Check if the selected edge is creating
		// a cycle or not (Cycle is created if u
		// and v belong to same set)
		if (set_u != set_v)
		{
			// Current edge will be in the MST
			// so print it
			//cout << u << " - " << v << endl;
            mst[v].push_back(u);
            mst[u].push_back(v);
			// Update MST weight
			mst_wt += it->first;

			// Merge two sets
			ds.merge(set_u, set_v);
		}
	}
    std::vector<int> roots;
    std::set<int> oriented_roots;
    for (int i = 0; i < V; i++){
        if (ds.parent[i] == i){
            orientSpanningTree(i, nx, ny, nz, mst);
            roots.push_back(i);
        }
        int root = ds.find(i);
        oriented_roots.insert(root);
    }
    std::cout << "roots:" << oriented_roots.size() << std::endl;
    std::cout << "roots1:" << roots.size() << std::endl;
    /*
    if (roots.size() > 1) {
        oriented_roots.insert(roots[0]);
        int oriented_roots_before = 0;
        while (oriented_roots.size() > oriented_roots_before) {
            oriented_roots_before = oriented_roots.size();
            for (int r = 1; r < roots.size(); r++) {
                int root = roots[r];
                if (!oriented_roots.count(root)) {
                    int smallestIndex = -1;
                    double min = 1e10;
                    for (int j = 0; j < k_use; j++) {
                        int n = knn[root * k + j];
                        if (n >= 0) {
                            int other_root = ds.find(n);
                            if (oriented_roots.count(other_root) > 0) {
                                double d = norm(root, n, x, y, z);
                                if (d < min){
                                    min = d;
                                    smallestIndex = n;
                                }
                            }
                        }
                    }
                    if (smallestIndex >= 0){
                        if (nx[root] * nx[smallestIndex] + ny[root] * ny[smallestIndex] + nz[root] * nz[smallestIndex] < 0){
                            reverseOrientation(root, ds, nx, ny, nz);
                        }
                        oriented_roots.insert(root);
                    }
                }
            }
        }
    }
    */
	return mst_wt;
}

// Driver program to test above functions
void orient_kruskal(int* knn, float* nx, float* ny, float* nz, float* x, float* y, float* z,
            int N, int k, int k_use){
	/* Let us create above shown weighted
	and undirected graph */
	int V = N;
	Graph g(V);
    std::set<std::pair<int, int>> mySet;
    std::unordered_map<long long int, std::pair<int, int>> uordmap;
    //uordmap.reserve(N * k_use);
    std::cout << "orienting" << std::endl;
    for (int i = 0; i < N; i++){
        for (int j = 0; j < k_use; j++){
            int n = knn[i * k + j];
            if (n >= 0){
                //if (i < n){
                auto pair = order(i, n);
                long long int key = N * pair.first + pair.second;
                auto success = uordmap.insert({key, pair});
                if (success.second){
                    double norm = nx[i] * nx[n] + ny[i] * ny[n] + nz[i] * nz[n];
                    double weight = std::max(0.0, 1.0 - abs(norm));
                    g.addEdge(pair.first, pair.second, weight);
                    //mySet.insert(order(i, n));
                }
            }
        }
    }
	// making above shown graph


	cout << "Edges of MST are \n";
	double mst_wt = g.kruskalMST(nx, ny, nz, knn, k, k_use, x, y, z);

	cout << "\nWeight of MST is " << mst_wt <<std::endl;

}


void orient(int* knn, float* nx, float* ny, float* nz, float* x, float* y, float* z,
            int N, int k, int k_use){
    /*std::cout << "orienting" << std::endl;
    //std::set<std::pair<int, int>> mySet;
    Graph* graph = createGraph(N);
    for (int i = 0; i < N; i++){
        std::cout << i << std::endl;
        for (int j = 0; j < k; j++){
            int n = knn[i * k + j];
            if (n >= 0){
                //std::cout << order(i, n).first << " " << order(i, n).second << std::endl;
                if (i < n){
                    //mySet.insert(order(i, n));
                    addEdge(graph, i, n, norm(i, n, x, y, z));
                }
            }
        }
    }*/
    std::cout << "Performing MST" <<std::endl;
    MSTOrientPointCloud(knn, nx, ny, nz, x, y, z, N, k, k_use);
    std::cout << "Ended!" << std::endl;

}