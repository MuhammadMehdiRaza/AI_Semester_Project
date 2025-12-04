def get_data(source_data: list[list[float]]) -> list[list[float]]:
    data_lists: list[list[float]] = []
    for data in source_data:
        for i, el in enumerate(data):
            if len(data_lists) < i + 1:
                data_lists.append([])
            data_lists[i].append(float(el))
    return data_lists
def calculate_each_score(
    data_lists: list[list[float]], weights: list[int]
) -> list[list[float]]:
    score_lists: list[list[float]] = []
    for dlist, weight in zip(data_lists, weights):
        mind = min(dlist)
        maxd = max(dlist)
        score: list[float] = []
        if weight == 0:
            for item in dlist:
                try:
                    score.append(1 - ((item - mind) / (maxd - mind)))
                except ZeroDivisionError:
                    score.append(1)
        elif weight == 1:
            for item in dlist:
                try:
                    score.append((item - mind) / (maxd - mind))
                except ZeroDivisionError:
                    score.append(0)
        else:
            msg = f"Invalid weight of {weight:f} provided"
            raise ValueError(msg)
        score_lists.append(score)
    return score_lists
def generate_final_scores(score_lists: list[list[float]]) -> list[float]:
    final_scores: list[float] = [0 for i in range(len(score_lists[0]))]
    for slist in score_lists:
        for j, ele in enumerate(slist):
            final_scores[j] = final_scores[j] + ele
    return final_scores
def procentual_proximity(
    source_data: list[list[float]], weights: list[int]
) -> list[list[float]]:
    data_lists = get_data(source_data)
    score_lists = calculate_each_score(data_lists, weights)
    final_scores = generate_final_scores(score_lists)
    for i, ele in enumerate(final_scores):
        source_data[i].append(ele)
    return source_data