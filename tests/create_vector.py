import random


def generate_vectors(start, end, lenght, num_vectors):
    vectors = []
    for i in range(num_vectors):
        vector = list(range(start+1 + i * lenght, start+1 + (i + 1) * lenght))
        vectors.append(vector)
    return vectors

def generate_vectors_same(start, end, lenght, num_vectors):
    vectors = []
    for i in range(num_vectors):
        vector = list(range(start+1, start+1 +lenght))
        vectors.append(vector)
    return vectors

def insert_prefix_vector(vectors, start, end, lenght):
    prefix_vector = list(range(start+1, start+1 +lenght))

    new_vectors = []
    for vector in vectors:
        new_vector = prefix_vector + vector
        new_vectors.append(new_vector)

    return new_vectors

def generate_mixed_vectors(start, end, length, num_vectors, identical_ratio=0.1):
    vectors = []
    identical_count = int(num_vectors * identical_ratio)  # 동일한 벡터의 수
    different_count = num_vectors - identical_count  # 다른 벡터의 수

    identical_vectors = generate_vectors_same(start, end, length, identical_count)
    vectors.extend(identical_vectors)

    different_vectors = generate_vectors(start, end, length, different_count)
    vectors.extend(different_vectors)

    random.shuffle(vectors)

    return vectors