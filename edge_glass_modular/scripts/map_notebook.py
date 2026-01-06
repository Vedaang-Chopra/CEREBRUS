
import json

NOTEBOOK_PATH = "/Users/vedaangchopra/all_data/complete_technical_work/all_projects_implemented/Edge Assistant/edge_glass_modular/notebooks/03_trm_vlm_qa_training_FIXED.ipynb"

def map_notebook():
    with open(NOTEBOOK_PATH, 'r') as f:
        nb = json.load(f)
    
    print(f"Total cells: {len(nb['cells'])}")
    code_cell_idx = 0
    
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            preview = source[:100].replace('\n', ' ')
            print(f"[{i}] Code Cell #{code_cell_idx}: {preview}...")
            
            if "optimizer =" in source or "AdamW" in source:
                print(f"    -> CONTAINS OPTIMIZER SETUP (Candidates for move)")
            if "DataLoader" in source:
                print(f"    -> CONTAINS DATALOADER (New location anchor)")
            if "collate_qa_batch" in source:
                print(f"    -> CONTAINS COLLATE FUNCTION")
                
            code_cell_idx += 1
        elif cell['cell_type'] == 'markdown':
             source = "".join(cell['source'])
             print(f"[{i}] Markdown: {source[:50].replace('\n', ' ')}...")

if __name__ == "__main__":
    map_notebook()
