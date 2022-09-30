# CORRER ALGORITMO ANTES PARA COMPILAR FUNCIONES

using CairoMakie
using Random
using StatsBase
using Printf
using MAT
using Serialization
using Dates
using StaticArrays
#using PyPlot
using DataFrames
using LinearAlgebra
#using CSV
#using BenchmarkTools
#using ColorSchemes
#using StaticArrays
#using FLoops

function sigmoid_temp(x::Float64, temp0::Float64, norm::Float64, rate::Float64)
    # norm: normalizing factor
    # normalizar con /(temp0 / 2)
    # rate: gradient degree
    return (-temp0/(1+exp( rate*(-temp0*x*norm + temp0) ))) + temp0
end

sigmoid_temp(x::Int, temp0, rate) = sigmoid_temp(Float64(x), temp0, rate)

function slow_temp(x::Float64, temp0::Float64, beta::Float64)
    return temp0 / (1 + beta * x)
end

# functions
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

function energy_total(matrix::Matrix{Float64})
    # get total energy
    #matrix = cyclic_matrix(matrix)
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

function run_loop(n_experiments::Int, matrix::Matrix{Float64}, T::Vector{Float64}, len::Int, kB::Float64)
    for ii = 1:n_experiments  - 1  
        shuffled_matrix = copy(matrix)

        shuffled_matrix, idx = shuffle_rows(shuffled_matrix)
        #println("AA)")

        # get indices
        idx_A = @SVector [mod(idx[1] - 1 - 1, len) + 1, idx[1], mod(idx[1], len) + 1]
        #println(idx[1])
        idx_B = @SVector [mod(idx[2] - 1 - 1, len) + 1, idx[2], mod(idx[2], len) + 1]
        

        # new - old
        delta_energy = energy_efficient(matrix, shuffled_matrix, idx_A, idx_B)

        #display(delta_energy)

        if delta_energy < 0
            matrix = shuffled_matrix

        elseif rand() < exp(-delta_energy/(kB * T[ii]))
            matrix = shuffled_matrix
        end
        #temp_vector[ii + 1] = slow_temp(temp_vector[ii],eta)
    end
    energy = energy_total(cyclic_matrix(matrix))
    
    #path_matrix[:,:,2] = matrix;
    return matrix, energy
end


function get_path(n_experiments::Int, matrix::Matrix{Float64}, temp0::Float64, norm_fact::Float64, sigmoid_rate::Float64)
    #n_experiments = 4000
    #matrices = zeros(size(matrix)[1], size(matrix)[2], n_experiments)
    # siempre iniciar con un random shuffle
    
    dim = size(matrix)
    
    matrix = shuffle_all(matrix);

    energy = 0.

    kB = 10.
    
    #sigmoid_rate = (n_experiments/2)^-1 # for symmetric curve
    sigmoid_norm = (n_experiments/norm_fact)^-1 # for asymmetric curve
    
    X = 1.:n_experiments
    
    temp = [sigmoid_temp(xi, 2., sigmoid_norm, .05) for xi in X];
    #temp = [slow_temp(xi, 1., 1.) for xi in X]
        
    #print(dim)
    matrix, energy = run_loop(n_experiments, matrix, temp, dim[1], kB)    
    #path_matrix[:,:,2] = matrix;
    return matrix, energy#, temp#, temp_vector, energy_vector
end

function run_experiments(n_experiments::Int, mat::Matrix{Float64}, k::Int, temp0::Float64, p1::Float64, p2::Float64)
    path = zeros(size(mat)[1], size(mat)[2], k)
    
    energ = zeros(k)
    
    for ii in 1:k
        path[:,:,ii], energ[ii] = get_path(n_experiments, mat, temp0, p1, p2);
    end
    
    energyyy, index = findmin(energ)
    
    @printf("E: %f", energyyy)

        
    return path[:,:,index], energyyy
end     


# regla: 
# 8  segundos para 10_000 iteraciones, 200 datos
# 15 segundos para 20_000 iteraciones, 200 datos
# 20 segundos para 25_000 iteraciones, 200 datos, sweetspot?
# 24 segundos para 30_000 iteraciones, 200 datos

# no multithreading strategy: long dataset gets no repetition, get the lowest

#@time path, energy = get_path(1_00_000, [x y], 3, 20., 3., 2.);
function main(filename::String, N::Int, k::Int)
    #str = readline()
    fileIn = matopen("./datasets/"*filename*".mat")
    data = read(fileIn)
    x = data["x"]# [1:200];
    y = data["y"]# [1:200];
    close(fileIn);
    path, energy = run_experiments(N, [x y], k, 10., 4., 8.);
    # cat output to clipboard
    clipboard(energy)

    serialize("./paths/"*filename*"_"*string(Int(ceil(energy)))*"_"*string(Dates.format(now(), "HH_MM_SS"))*".dat",path);
    return path, energy
end

#=
figure = Figure(resolution=(800,500))

#NO PLOTEES AMBOS EN UN MISMO PLOT
colors = (1:length(x) + 1)
alpha = ones(length(colors)) * 0.5
cmap = :davos

axis1 = Makie.Axis(figure[1,1], title="Bad path, E(end): "*string(round(energy_total([x y]), digits=3)) )
scatter!(axis1, cyclic_matrix(shuffle_all([x y])), color = colors, colormap=cmap)
lines!(axis1, cyclic_matrix(shuffle_all([x y])), color = colors, colormap=cmap)

axis2 = Makie.Axis(figure[1,2], title="Optimized path, E(end): "*string(round(energy, digits=3)) )
scatter!(axis2, cyclic_matrix(path), color = colors, colormap=cmap)
lines!(axis2, cyclic_matrix(path), color = colors, colormap=cmap)

figure
=#
