def map_gaze_to_section(gaze_point, screen_config):
    """
    Map a gaze point (x, y) to a section ID based on the screen configuration.

    screen_config should include:
      - screen_width: total width of the screen in pixels
      - screen_height: total height of the screen in pixels
      - rows: number of rows in the grid
      - cols: number of columns in the grid
    """
    x, y = gaze_point
    screen_width = screen_config["screen_width"]
    screen_height = screen_config["screen_height"]
    rows = screen_config["rows"]
    cols = screen_config["cols"]
    
    cell_width = screen_width / cols
    cell_height = screen_height / rows
    
    # Determine column and row indices (0-indexed)
    col_index = int(x // cell_width)
    row_index = int(y // cell_height)
    
    # Clamp indices to grid dimensions
    if col_index >= cols:
        col_index = cols - 1
    if row_index >= rows:
        row_index = rows - 1
    
    # Compute section ID in row-major order (1-indexed)
    section_id = row_index * cols + col_index + 1
    return section_id
