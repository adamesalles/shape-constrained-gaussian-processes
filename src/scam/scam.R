library(scam)
library(dplyr)
library(readr)

# Function to remove points equally spaced in the dataset
remove_points <- function(x, y, nobs) {
  idx <- seq(1, length(x), length.out = nobs)
  return(list(x = x[idx], y = y[idx]))
}

# Paths to datasets
data_paths <- list(
  "uniform_new_HC" = "../../data/Gaussian_logCA0_uniform_J=20_HC.csv",
  "uniform_new_MC" = "../../data/Gaussian_logCA0_uniform_J=20_MC.csv",
  "adaptive_new_HC" = "../../data/Gaussian_logCA0_adaptive_J=20_HC.csv",
  "adaptive_new_MC" = "../../data/Gaussian_logCA0_adaptive_J=20_MC.csv"
)

# Define nobs scenarios
nobs_scenarios <- append(seq(6, 20, by = 2), 21)

# Data frame to store results
results <- data.frame(model = character(), scenario = integer(), dataset = character(), mse = numeric())

x_true <- read_csv("../../data/true_Gaussian_logCA0_HC.csv")$a0
# Experiment loop
for (dataset_name in names(data_paths)) {
  dataset_path <- data_paths[[dataset_name]]
  
  # Read data
  data <- read_csv(dataset_path)
  x <- data$a0
  y <- data$lc_a0
  
  
  for (nobs in nobs_scenarios) {
    
    # Remove points to get desired number of observations
    sampled_data <- remove_points(x, y, nobs)
    x_sampled <- sampled_data$x
    y_sampled <- sampled_data$y
    
    # Fit the SCAM model
    model <- tryCatch(
      {
        scam(y_sampled ~ s(x_sampled, bs = "cx"))
      },
      error = function(e) {
        cat("Error in model fitting:", e$message, "\n")
        return(NULL)
      }
    )
    
    if (!is.null(model)) {
      # Generate predictions on a regular grid from 0 to 1
      grid <- data.frame(x_sampled = x_true)
      grid$y_pred <- predict(model, newdata = grid)
      # print("Got here")
      
      # Save predictions
      output_file <- paste0("../../experiments/", dataset_name, "/SCAM_", nobs , "_nobs_",  ".csv")
      write.csv(grid, output_file, row.names = FALSE)
      
      # Calculate Mean Squared Error (MSE) on the sampled data
      #y_pred_sampled <- predict(model, newdata = data.frame(x = x_sampled))
      #mse <- mean((y_sampled - y_pred_sampled)^2)
      
      # Store results
      #results <- rbind(
      #  results,
      #  data.frame(model = "SCAM", scenario = nobs, dataset = dataset_name, mse = mse)
      #)
      
      cat("Finished SCAM model on", dataset_name, "with nobs =", nobs, "\n")
      rm(grid)
    }
  }
}

# Save MSE results to CSV
#write.csv(results, "scam_mse_results.csv", row.names = FALSE)

# Display results
#print(results)


ggplot(data, aes(x = x, y = y)) +
  geom_point(color = "blue", alpha = 0.5) +             # Original data points
  geom_line(data = grid, aes(x = x, y = y_pred), color = "red") +  # Predicted line on grid
  labs(title = "Shape-Constrained Additive Model Fit",
       x = "x", y = "y") +
  theme_minimal()
