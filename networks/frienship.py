from collections import Counter

users = [
    {"id": 0, "name": "Hero"},
    {"id": 1, "name": "Dunn"},
    {"id": 2, "name": "Sue"},
    {"id": 3, "name": "Chi"},
    {"id": 4, "name": "Thor"},
    {"id": 5, "name": "Clive"},
    {"id": 6, "name": "Hicks"},
    {"id": 7, "name": "Devin"},
    {"id": 8, "name": "Kate"},
    {"id": 9, "name": "Klein"}
]
friendships = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4),
               (4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)]
for user in users:
    user["friends"] = []

for i, j in friendships:
    users[i]["friends"].append(users[j])  # add i as a friend of j
    users[j]["friends"].append(users[i])  # add j as a friend of i


def number_of_friends(user_name):
    """
    How many friends does a user have ?

    """
    for user_ in users:
        if user_['name'] == user_name:
            return len(user_["friends"])


def total_connections():
    conns = 0
    for user_ in users:
        conns += number_of_friends(user_['name'])
    return conns


print total_connections()

# sort users by number of friends they have
num_of_friends_by_id = [(user['id'], number_of_friends(user['name'])) for user in users]
print sorted(num_of_friends_by_id, key=lambda (user_id, num_friends): num_friends, reverse=True)


def get_foaf(user_):
    return [foaf['id'] for friend_ in user_['friends'] for foaf in friend_['friends']]


print get_foaf(users[0])


# Mutual Friends
def not_the_same(user_, other_user):
    """
    two users are not the same if they have different ids
    """
    return user_['id'] != other_user['id']


def not_friends(user_, other_user):
    return all(not_the_same(friend, other_user) for friend in user_['friends'])


def mutual_friends(user_):
    return Counter(foaf["id"]
                   for friend in user_["friends"] for foaf in friend["friends"] if
                   not_the_same(user_, foaf) and not_friends(user_, foaf))


print mutual_friends(users[3])























