#using CairoMakie
using StatsBase, .Threads, SIMD, MAT, StaticArrays, LinearAlgebra, BenchmarkTools, Random, CSV, DataFrames
using CairoMakie
using Printf

function energy_total(matrix::Matrix{Float64})
  # get total energy
  matrix = cyclic_matrix(matrix)
  dist = get_distance(matrix)
  # display(dist)
  return sum(dist[diagind(dist, 1)])
end

function energy_efficient(normal::Matrix{Float64},shuffled::Matrix{Float64},idx_1::SVector{3, Int64}, idx_2::SVector{3, Int64})
  # get energy between points
  # faster in memory, i guesss
  
  #shuffled, idx = shuffle_rows(mat1);
  A1 = get_distance(normal[idx_1,:])
  A2 = get_distance(normal[idx_2,:])
  B1 = get_distance(shuffled[idx_1,:])
  B2 = get_distance(shuffled[idx_2,:])
  
  E_normal = sum(A1[diagind(A1, 1)]) + sum(A2[diagind(A2, 1)])
  E_shuffled = sum(B1[diagind(B1, 1)]) + sum(B2[diagind(B2, 1)])
  return E_shuffled - E_normal
end

function cyclic_matrix(matrix::Matrix{Float64})
  # returns matrix with repeated starting point
  # pal plot na más
  return [matrix; matrix[1,:]']; # que regrese al inicio
end

function shuffle_all(matrix::Matrix{Float64})
  return matrix[shuffle(1:end), :]
end

function shuffle_rows(matrix::Matrix{Float64})
  N = size(matrix, 1)

  rows = sample(1:N, 2, replace=false)

  matrix[rows[1],:] .= matrix[rows[1],:] + matrix[rows[2], :]
  matrix[rows[2],:] .= matrix[rows[1],:] - matrix[rows[2], :]
  matrix[rows[1],:] .= matrix[rows[1],:] - matrix[rows[2], :]
  
  return matrix, rows
end

function get_distance(matrix::Matrix{Float64})
  x = matrix[:,1]
  y = matrix[:,2]
  distx = triu(x' .- x);
  #display(distx.^2)
  disty = triu(y' .- y);
  
  return  sqrt.(disty.^2 + distx.^2);
end


function sigmoid_temp(x::Float64, temp0::Float64, norm::Float64, rate::Float64)
  # norm: normalizing factor
  # normalizar con /(temp0 / 2)
  # rate: gradient degree
  return (-temp0/(1+exp( rate*(-temp0*x*norm + temp0) ))) + temp0
end

function get_path_vectors(n_experiments::Int, matrix::Matrix{Float64}, temp0::Float64, eta::Float64)
  #n_experiments = 4000
  
  dim = size(matrix)
  
  matrices = zeros(dim[1], dim[2], n_experiments)
  # siempre iniciar con un random shuffle
  matrix = shuffle_all(matrix);

  #temp0 = 10
  #temp_vector = zeros(n_experiments)
  #temp_vector[1] = temp0
  
  energy_vector = zeros(n_experiments)
  energy_vector[1] = energy_total(matrix)

  final_temp = 0.0

  #eta = 0.01

  path_matrix = zeros(dim[1], dim[2], 2)
  #path_matrix[:,:,1] = matrix

  #matrix = path_matrix[:,:,1]
  matrices[:,:,1] = matrix

  kB = 10
  T = 10
  
  for ii = 1:n_experiments - 1
      
      len = dim[1]
      
      shuffled_matrix = copy(matrix)

      shuffled_matrix, idx = shuffle_rows(shuffled_matrix)
      
      # get indices
      idx_A = @SVector [mod(idx[1] - 1 - 1, len) + 1, idx[1], mod(idx[1], len) + 1]
      idx_B = @SVector [mod(idx[2] - 1 - 1, len) + 1, idx[2], mod(idx[2], len) + 1]
      
      # new - old
      delta_energy = energy_efficient(matrix, shuffled_matrix, idx_A, idx_B)

      #display(delta_energy)

      if delta_energy < 0
          matrix = shuffled_matrix

      elseif rand() < exp(-delta_energy/(kB * temp0 / (ii + 1)))
          matrix = shuffled_matrix
      end
      #temp_vector[ii + 1] = slow_temp(temp_vector[ii],eta)
      #sigmoid works betterç
      #evita que usemos un if break para la temp lineal
      #temp_vector[ii + 1] = sigmoid_temp(ii, temp0, eta)
      #display(temp_vector[ii+1])
      energy_vector[ii + 1] = energy_total(matrix)
      #temp0::Float64, temp::Float64, beta::Float64, iter::Int)
      matrices[:,:,ii+1] = matrix
  end
  #path_matrix[:,:,2] = matrix;
  return matrices, energy_vector
end

function main(N)
  data = CSV.read("./data/qatar.csv",DataFrame)
  x = data.x
  y = data.y
  t0 = 20.;
  eta = 0.1;

  #N = 10_000;
  #X, E  = get_path_vectors(N, [x y], t0, eta)
  return get_path_vectors(N, [x y], t0, eta)  
end

function animate(arr, energ, N,STEP::Int64)
	set_theme!(theme_black())
	colors = (1:size(arr,1) + 1)
	#alpha = ones(length(colors)) * 0.5
	cmap = :buda
  f = Figure(resolution=(800,800)) 
  ax = Axis(f[1,1])
  ax.aspect=DataAspect()
  #rowsize!(f.layout, 1, ax.scene.px_area[].widths[2]) # set colorbar height
  #lines!(ax,cyclic_matrix(path),color=collect(colors),colormap=cmap)
  #framerate = 30
  #record(f, "tsp.mp4", 1:STEP:N;
  #        framerate = framerate) do idx
  #    ln = 	lines!(ax,cyclic_matrix(arr[:,:,idx]))    
  #end
  counter = 0.
  for ii in 1:STEP:N
    ln=lines!(ax,cyclic_matrix(arr[:,:,ii]), color=colors, colormap=cmap)
    sc=scatter!(ax,cyclic_matrix(arr[:,:,ii]), color=colors, colormap=cmap)
    tx=text!(ax, 26000, 51600, text="E="*@sprintf("%0.3f", energ[ii]))
    save("./images/movie"*@sprintf("%03i",counter)*".png", f)
    counter += 1.
    delete!(ax.scene, ln)
    delete!(ax.scene, sc)
    delete!(ax.scene, tx)
  end
  run(`ffmpeg -framerate 30 -pattern_type glob -i './images/*.png' -c:v libx264 ./videos/out.mp4 -y`)


end