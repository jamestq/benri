import pandas as pd
import numpy as np
import tqdm
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
import h5py
import typer

app = typer.Typer()

def get_distance_matrix(df: pd.DataFrame, embedding_column="embedding") -> None:
    """Compute the distance matrix for the embeddings

    Args:
        df (pd.DataFrame): DataFrame containing the embedding
        embedding_column: the column containing the embedding values

    """
    embeddings = np.stack(df[embedding_column].values) #type: ignore
    distvec = pdist(embeddings, metric='euclidean')    
    index = df.index.to_list()    
    with h5py.File('dist_matrix.hdf5', 'w') as f:
        f.create_dataset('distmatrix', data=squareform(distvec))
        f.create_dataset('index', data=np.array(index))    


def farthest_first(selected_ids: list[int] = [], k: int = 400) -> list[str]:
    """Perform farthest-first traversal to select k diverse samples.

    Args:
        selected (list[str]): List of initially selected identifiers. If 0, start with a random sample.
        k (int): Number of samples to select.
    """
    with h5py.File("dist_matrix.hdf5", "r") as f:
        dist_matrix = f["distmatrix"][:] # type: ignore
        index = f["index"][:]    # type: ignore
    if len(selected_ids) == 0:
        np.random.seed(42)
        selected_ids = [np.random.randint(0, index.shape[0])]     # type: ignore
    for _ in tqdm.tqdm(range(k)):                
        if len(selected_ids) == 1:
            distances = dist_matrix[selected_ids[-1]] # type: ignore - Get the only element in the selected_ids array
            next_id = np.argmax(distances) # type: ignore
            selected_ids.append(next_id)
        else:
            candidates = list()            
            for id in range(index.shape[0]): #type: ignore
                if id in selected_ids:
                    continue                                                            
                distances = dist_matrix[id, selected_ids] #type: ignore
                min_distance = np.min(distances) #type: ignore               
                candidates.append((id, min_distance))            
            next_id = max(candidates, key=lambda x: x[1])[0]
            selected_ids.append(next_id)
    actual_ids = [index[id] for id in selected_ids] #type: ignore
    return actual_ids #type: ignore

@app.command()
def main(
    dffile: Path,
    output: Path = Path("output.txt"),
    calculate_dist_matrix: bool = False
):
    df = pd.read_parquet(dffile)
    if calculate_dist_matrix:
        get_distance_matrix(df)
    selected_ids = farthest_first()
    selected = df[df.index.isin(selected_ids)]["path"].to_list()
    Path(output).write_text("\n".join(selected))

if __name__ == "__main__":
    app()