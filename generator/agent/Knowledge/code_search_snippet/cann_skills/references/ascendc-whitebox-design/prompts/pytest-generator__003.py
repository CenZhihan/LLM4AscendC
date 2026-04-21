if p.get("ndim", 2) == 2:
    shape = (p["inDimx"], p["x_last"])
elif p["ndim"] == 3:
    shape = (batch, seq, p["x_last"])  # inDimx = batch * seq
elif p["ndim"] == 4:
    shape = (batch, n, s, p["x_last"])
