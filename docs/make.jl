using Turbulox
using Documenter

DocMeta.setdocmeta!(Turbulox, :DocTestSetup, :(using Turbulox); recursive = true)

makedocs(;
    modules = [Turbulox],
    authors = "Syver Døving Agdestein <syverda@gmail.com> and contributors",
    sitename = "Turbulox.jl",
    format = Documenter.HTML(;
        canonical = "https://agdestein.github.io/Turbulox.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = ["Home" => "index.md"],
)

deploydocs(; repo = "github.com/agdestein/Turbulox.jl", devbranch = "main")
