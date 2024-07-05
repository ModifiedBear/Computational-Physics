using GLMakie
using LinearAlgebra
using StatsBase
using Dates



function apply_boundaries(mat, nx, ny)
    # remember mat[i,j] = mat[x, y]
    M = copy(mat)
    M[:,1] .= 0;
    M[1,:] .= 0;
    M[:,end] .= 0;
    M[end,:] .= 0;
    
    width_x = round(Int, nx*0.4)
    width_y = width_x
    #M[width_x + 1:(end-width_x), width_y+1:(end-width_y)] .= 2
    
    # capacitor
    M[width_x + 1, width_y:(end-width_y)] .= 1
    M[end - width_x + 1, width_y:(end-width_y)] .= -1
    
    # particles
    #M[width_x, width_y] = -1;
    #M[end-width_x,end-width_y] = 1;
    #M[15, 45] = 1;
    #M[80, 31] = -1;
    
    
    return M
end

function random_charges(mat, nx, ny, coord)

    # random particles
    # sample coordinates
    
    q = rand((1,-1), 10) ;
    mat[coord] = q;
    return mat
end
    
function relaxation(mat, nx, ny, N, tol)
    # N: number of iterations
    coord = sample(LinearIndices(mat), 10); # pick points
    for ii = 1:N
        mat_last = copy(mat);
        mat[2:end-1, 2:end-1] .= 0.25 * 
        (
            mat[3:end,2:end-1] + 
            mat[1:end-2, 2:end-1] + 
            mat[2:end-1, 3:end] +
            mat[2:end-1, 1:end-2]
        );
        mat = apply_boundaries(mat, nx, ny);
        #mat = random_charges(mat, nx, ny, coord)
        if maximum(norm.(mat_last - mat)) < tol
            println("Tol achieved");
            break
        end
    end
    # apply boundary
    
    return mat;
end

begin
    dx, dy = 0.1,0.1;
    xs, ys = -1:dx:1, -1:dy:1;
    nx = length(xs);
    ny = length(ys);
    N = 5000;
    V = zeros(nx, ny);
    
    V = zeros(nx, ny);
    V = apply_boundaries(V, nx, ny);
    V = relaxation(V, nx, ny, N, 0.0001);
    Ex = -diff(V,dims=1)[:,1:end-1]; Ey = -diff(V, dims=2)[1:end-1,:];
    k = 5;
    lengths=sqrt.(Ex.^ 2 .+ Ey .^ 2);

    # normalize E field
    Ex = Ex ./ lengths;
    Ey = Ey ./ lengths;

    set_theme!(theme_black())
    fig = Figure(resolution=(1000,800))
    ax = [Axis(fig[1,1], xlabel="x",ylabel="y"), 
          Axis3(fig[1,2], 
            elevation = 0.15pi, 
            azimuth = -0.25pi,
            zlabel="V",
            aspect=(2,2,3)),
          Axis(fig[1,3],xlabel="x", ylabel="y")]
    #hm=heatmap!(ax[1], xs, ys,V; colormap=Reverse(:roma))

    #sf=surface!(ax[2], xs, ys,V; colormap=Reverse(:roma),shading = true)
    ct=contour!(ax[1], xs, ys, V; levels=-0.6:0.07:0.6, color=:white)
    #ct3=contour3d!(ax[2], xs, ys, V; levels=-0.5:0.07:0.5, color=:white)
    wireframe!(ax[2], xs, ys, V,color=:white,shininess=0,linewidth=1.)


    ax[1].aspect = DataAspect()
    #cm=Colorbar(fig[1,3], hm; label=L"\phi /\text{Volts}", tellheight=true)
    rowsize!(fig.layout, 1, ax[1].scene.px_area[].widths[2]) # set colorbar height
    save("FEM-2"*string(now())*".png", fig)
    fig
    
end
