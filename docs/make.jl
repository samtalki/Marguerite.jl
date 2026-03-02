using Marguerite
using Documenter

DocMeta.setdocmeta!(Marguerite, :DocTestSetup, :(using Marguerite); recursive=true)

makedocs(;
    modules=[Marguerite],
    warnonly=[:missing_docs],
    authors="samtalki <saimouer@gmail.com> and contributors",
    sitename="Marguerite.jl",
    format=Documenter.HTML(;
        canonical="https://samtalki.github.io/Marguerite.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "Oracles" => "oracles.md",
        "Differentiation" => "differentiation.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/samtalki/Marguerite.jl",
    devbranch="main",
)
