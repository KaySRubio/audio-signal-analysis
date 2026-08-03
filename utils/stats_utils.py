
# Function that will print a nice table of means and standard deviation by group, sorted rows
def print_mean_and_std_by_group(*, df, groupby, variable, round = 3):
  summary = (
    df.groupby(groupby)
      .agg({
        variable: ["mean", "std"],
      })
  )

  # Flatten the column names
  summary.columns = [f"{variable}_mean", f"{variable}_std"]

  # Sort by the mean
  summary = summary.sort_values(f"{variable}_mean")

  print(summary.round(round))