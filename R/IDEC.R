IDEC <- R6::R6Class("IDEC",
                    inherit = DEC,
                    lock_objects = FALSE,
                    public = list(
                      gamma = 0.1,
                      
                      initialize = function(dims, n_clusters, alpha = 1.0, gamma = 0.1, initializer = 'glorot_uniform') {
                        super$initialize(dims = dims, n_clusters = n_clusters, alpha = alpha, initializer = initializer)
                        self$gamma <- gamma
                        
                        # rebuild IDEC model with two outputs: clustering and decoder
                        clustering_output <- self$model$output
                        decoder_output <- self$autoencoder$output
                        self$model <- keras_model(inputs = self$encoder$input, outputs = list(clustering_output, decoder_output))
                      },
                      
                      compile = function(optimizer = 'sgd', loss_weights = NULL) {
                        loss_weights <- loss_weights %||% list(1, self$gamma)
                        self$model$compile(
                          optimizer = optimizer,
                          loss = list('kld', 'mse'),
                          loss_weights = loss_weights
                        )
                      },
                      
                      load_encoder_weights = function(weights_path) {
                        self$autoencoder$load_weights(weights_path)
                        cat("Autoencoder weights loaded into IDEC model.\n")
                      },
                      
                      fit = function(x, y = NULL, maxIter = 2e4, batchSize = 256L, 
                                     tolerance = 1e-4, updateInterval = 500, saveDir = NULL) {
                        km <- kmeans(self$encoder$predict(x), centers = self$n_clusters, nstart = 20)
                        previousPrediction <- km$cluster
                        self$model$get_layer(name = 'clustering')$set_weights(list(km$centers))
                        
                        q <- self$model$predict(x, verbose = 0)[[1]]
                        p <- self$targetDistribution(q)
                        index <- 0
                        indexArray <- 1:nrow(x)
                        loss <- 0
                        
                        for (i in seq_len(maxIter)) {
                          if (i %% updateInterval == 1) {
                            q <- self$model$predict(x, verbose = 0)[[1]]
                            p <- self$targetDistribution(q)
                            currentPrediction <- max.col(q)
                            deltaLabel <- sum(currentPrediction != previousPrediction) / length(currentPrediction)
                            
                            if (!is.null(y)) {
                              acc <- ACC(y, currentPrediction)
                              nmi <- NMI(y, currentPrediction)
                              cat(sprintf("Itr %d: deltaLabel = %.5f, ACC = %.5f, NMI = %.5f\n",
                                          i, deltaLabel, acc, nmi))
                            } else {
                              cat(sprintf("Itr %d: deltaLabel = %.5f\n", i, deltaLabel))
                            }
                            
                            if (i > 1 && deltaLabel < tolerance) {
                              message("Reached tolerance threshold. Stopping training...")
                              break
                            }
                            
                            previousPrediction <- currentPrediction
                          }
                          
                          if (i %% 50 == 0) {
                            cat(sprintf("Itr %d: loss = %.5f\n", i, loss[[1]]))
                            training_metrics$loss <- c(training_metrics$loss, loss[[1]])  # Track loss if needed
                          }
                          
                          batch_start <- index * batchSize + 1
                          batch_end <- min((index + 1) * batchSize, nrow(x))
                          
                          if (batch_start > batch_end) {
                            index <- 0
                            next
                          }
                          
                          batchIndices <- indexArray[batch_start:batch_end]
                          loss <- self$model$train_on_batch(
                            x = x[batchIndices, ], 
                            y = list(p[batchIndices, ], x[batchIndices, ])
                          )
                          
                          index <- if ((index + 1) * batchSize <= nrow(x)) index + 1 else 0
                        }
                        
                        return(currentPrediction)
                      },
                      
                      save_autoencoder_weights = function(filepath = "autoencoder_pretrained.weights.h5") {
                        self$autoencoder$save_weights(filepath)
                        cat(sprintf("Autoencoder weights saved to %s\n", filepath))
                      }
                    )
)

mnist <- load_mnist()
x <- mnist$x
y <- mnist$y
n_clusters <- length(unique(y))

model <- IDEC$new(dims = c(784, 500, 500, 2000, 10), n_clusters = n_clusters)
model$pretrain(x, y = y)

model$compile()
y_pred <- model$fit(x, y = y)
