using EvidentialDeepLearning, Documenter, Literate

DocMeta.setdocmeta!(EvidentialDeepLearning, :DocTestSetup, :(using EvidentialDeepLearning); recursive=true)

inputdir = joinpath(@__DIR__, "..", "examples")
outputdir = joinpath(@__DIR__, "src", "examples")
mkpath(outputdir)
mkpath(joinpath(outputdir, "examples"))
for filename in readdir(inputdir)
    inpath = joinpath(inputdir, filename)
    cp(inpath, joinpath(outputdir, "examples", filename), force = true)
    Literate.markdown(inpath, outputdir; documenter = true)
end
cp(joinpath(@__DIR__, "..", "README.md"), joinpath(@__DIR__, "src", "index.md"), force=true)

makedocs(;
    modules=[EvidentialDeepLearning],
    authors="Jan Weidner <jw3126@gmail.com> and contributors",
    repo="https://github.com/jw3126/EvidentialDeepLearning.jl/blob/{commit}{path}#{line}",
    sitename="EvidentialDeepLearning.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://jw3126.github.io/EvidentialDeepLearning.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Tutorials" => ["Hello World" => "examples/hello_world.md",],
        # "Explanation" => [],
        # "Reference" => [],
        # "How-to guides" => [],
    ]
)

deploydocs(;
    repo="github.com/jw3126/EvidentialDeepLearning.jl",
    push_preview=true,
)
