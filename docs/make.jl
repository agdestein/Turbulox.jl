using Turbulox
using Documenter
using DocumenterCitations

DocMeta.setdocmeta!(Turbulox, :DocTestSetup, :(using Turbulox); recursive = true)

bib = CitationBibliography(joinpath(@__DIR__, "references.bib"))

makedocs(;
    modules = [Turbulox],
    authors = "Syver Døving Agdestein <syverda@gmail.com> and contributors",
    sitename = "Turbulox.jl",
    format = Documenter.HTML(;
        canonical = "https://agdestein.github.io/Turbulox.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = ["Home" => "index.md", "References" => "references.md"],
    plugins = [bib],
)

deploydocs(; repo = "github.com/agdestein/Turbulox.jl", devbranch = "main")
