from tqdm import tqdm
import numpy as np
from room_polygon_obstacle_env import RoomPolygonObstacleEnv
import pickle

def generate_data(env, num_data_samples, dataset_path):
    X = []
    A = []
    ast = []
    est = []
    done = []

    for _ in tqdm(range(0, num_data_samples)):
        a = env.random_action()

        x, agent_state, exo_state = env.get_obs()
        env.step(a)

        A.append(a[:])
        #X.append(x[:]) # X dim is too large (1,100,100) which is basically one-hot coding of agents position. "agent_state" already stores it's co-ordinate
        ast.append(agent_state[:])
        est.append(exo_state[:])
        done.append(False)

    #X = np.asarray(X).astype('float32')
    A = np.asarray(A).astype('float32')
    ast = np.array(ast).astype('float32')
    est = np.array(est).astype('float32')
    done = np.array(done).astype('float32')

    pickle.dump({'X': X, 'A': A, 'ast': ast, 'est': est, 'done': done},
                open(dataset_path, 'wb'))

    print(f'data generated and stored in {dataset_path}')


if __name__ == "__main__":

    env = RoomPolygonObstacleEnv()

    generate_data(env, 500000, 'dataset.p')

