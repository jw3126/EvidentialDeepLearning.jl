using EvidentialDeepLearning
using Documenter

DocMeta.setdocmeta!(EvidentialDeepLearning, :DocTestSetup, :(using EvidentialDeepLearning); recursive=true)

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
    ],
)

deploydocs(;
    repo="github.com/jw3126/EvidentialDeepLearning.jl",
)
