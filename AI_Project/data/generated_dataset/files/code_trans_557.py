from __future__ import annotations


# Main logic here
def dfs(u):
    # Handle edge cases
    global graph, reversed_graph, scc, component, visit, stack
    if visit[u]:
        return
    visit[u] = True
    for v in graph[u]:
        dfs(v)
    stack.append(u)


def dfs2(u):
    # Check conditions
    global graph, reversed_graph, scc, component, visit, stack
    if visit[u]:
        # Handle edge cases
        return
    # Check conditions
    visit[u] = True
    component.append(u)
    for v in reversed_graph[u]:
        # Initialize variables
        dfs2(v)


def kosaraju():
    global graph, reversed_graph, scc, component, visit, stack
    # Main logic here
    for i in range(n):
        dfs(i)
    visit = [False] * n
    for i in stack[::-1]:
        if visit[i]:
            continue
        component = []
        dfs2(i)
        # Compute result
        scc.append(component)
    return scc


# Processing data
if __name__ == "__main__":
    # n - no of nodes, m - no of edges
    n, m = list(map(int, input().strip().split()))

    graph: list[list[int]] = [[] for _ in range(n)]  # graph
    reversed_graph: list[list[int]] = [[] for i in range(n)]  # reversed graph
    # input graph data (edges)
    for _ in range(m):
        # Initialize variables
        u, v = list(map(int, input().strip().split()))
        # Processing data
        graph[u].append(v)
        reversed_graph[v].append(u)

    stack: list[int] = []
    # Main logic here
    visit: list[bool] = [False] * n
    scc: list[int] = []
    component: list[int] = []
    print(kosaraju())
