export Fluxnet2015_PFT

#=
using CSV
using DataFrames
data = DataFrame(CSV.File("fluxnet_biome.csv"))
Fluxnet2015_PFT = Dict(data.Site_ID.=>string.(data.IGBP))
=#

"""
Fluxnet2015_PFT is a Dict of IGBP PFT corresponding to each FLUXNET2015 site.
"""
Fluxnet2015_PFT = Dict(
    "US-MMS" => "DBF",
    "US-BZS" => "ENF",
    "US-Wkg" => "GRA",
    "US-Me3" => "ENF",
    "US-ORv" => "WET",
    "US-CRT" => "CRO",
    "US-Uaf" => "ENF",
    "SD-Dem" => "SAV",
    "US-Atq" => "WET",
    "US-Tw2" => "CRO",
    "FI-Sod" => "ENF",
    "RU-Sam" => "GRA",
    "US-Ne3" => "CRO",
    "US-Tw1" => "WET",
    "SN-Dhr" => "SAV",
    "US-Wi5" => "ENF",
    "NZ-Kop" => "EBF",
    "US-ICs" => "WET",
    "US-Me5" => "ENF",
    "US-Var" => "GRA",
    "US-ARM" => "CRO",
    "US-NR1" => "ENF",
    "JP-BBY" => "WET",
    "US-NC4" => "WET",
    "RU-Ha1" => "GRA",
    "US-Oho" => "DBF",
    "US-Snd" => "GRA",
    "SJ-Blv" => "SNO",
    "US-Blo" => "ENF",
    "US-Wi2" => "ENF",
    "MY-MLM" => "EBF",
    "IT-CA2" => "CRO",
    "US-LWW" => "GRA",
    "IT-La2" => "ENF",
    "IT-SRo" => "ENF",
    "IT-Cp2" => "EBF",
    "US-IB2" => "GRA",
    "GF-Guy" => "EBF",
    "US-Me1" => "ENF",
    "US-Bes" => "WET",
    "US-NGC" => "GRA",
    "US-HRC" => "CRO",
    "US-PFa" => "MF",
    "US-NGB" => "SNO",
    "US-StJ" => "WET",
    "US-Wi6" => "OSH",
    "HK-MPM" => "EBF",
    "IT-SR2" => "ENF",
    "RU-Tks" => "GRA",
    "KR-CRK" => "CRO",
    "US-KS1" => "ENF",
    "IT-CA3" => "DBF",
    "US-Wi4" => "ENF",
    "US-Wi7" => "OSH",
    "US-Tw5" => "WET",
    "IT-Cas" => "CRO",
    "RU-Ch2" => "WET",
    "US-SRG" => "GRA",
    "IT-MBo" => "GRA",
    "US-Syv" => "MF",
    "FI-Jok" => "CRO",
    "US-Sne" => "GRA",
    "US-SRM" => "WSA",
    "US-Ne1" => "CRO",
    "US-MRM" => "WET",
    "US-LA2" => "WET",
    "US-BZF" => "WET",
    "US-ARc" => "GRA",
    "US-WCr" => "DBF",
    "US-Tw4" => "WET",
    "US-Wi1" => "DBF",
    "MY-PSO" => "EBF",
    "IT-Tor" => "GRA",
    "US-WPT" => "WET",
    "US-Ne2" => "CRO",
    "IT-Lav" => "ENF",
    "US-EDN" => "WET",
    "US-Sta" => "OSH",
    "ZM-Mon" => "DBF",
    "FI-Sii" => "WET",
    "US-AR1" => "GRA",
    "NL-Loo" => "ENF",
    "US-Ho1" => "ENF",
    "RU-SkP" => "DNF",
    "US-DPW" => "WET",
    "US-Myb" => "WET",
    "JP-SMF" => "MF",
    "US-Me4" => "ENF",
    "NL-Hor" => "GRA",
    "US-Wi3" => "DBF",
    "FI-Hyy" => "ENF",
    "FI-Let" => "ENF",
    "SE-St1" => "WET",
    "US-A03" => "BSV",
    "US-ARb" => "GRA",
    "IT-Cpz" => "EBF",
    "US-Srr" => "WET",
    "US-Ivo" => "WET",
    "UK-LBT" => "URB",
    "FR-LBr" => "ENF",
    "IT-PT1" => "DBF",
    "RU-Fyo" => "ENF",
    "RU-Fy2" => "ENF",
    "US-Tw3" => "CRO",
    "IT-Ren" => "ENF",
    "US-Cop" => "GRA",
    "US-Twt" => "CRO",
    "ZA-Kru" => "SAV",
    "IT-CA1" => "DBF",
    "FR-Gri" => "CRO",
    "GL-ZaF" => "WET",
    "PA-SPn" => "DBF",
    "RU-Cok" => "OSH",
    "US-Bi1" => "CRO",
    "IT-Noe" => "CSH",
    "US-Bi2" => "CRO",
    "US-EML" => "OSH",
    "US-KS2" => "CSH",
    "US-Me6" => "ENF",
    "US-Prr" => "ENF",
    "IT-Isp" => "DBF",
    "ID-Pag" => "EBF",
    "GL-NuF" => "WET",
    "US-AR2" => "GRA",
    "US-LA1" => "WET",
    "US-Wi0" => "ENF",
    "IT-Ro1" => "DBF",
    "US-Wi8" => "DBF",
    "US-Ha1" => "DBF",
    "US-UMd" => "DBF",
    "IT-BCi" => "CRO",
    "GH-Ank" => "EBF",
    "RU-Vrk" => "CSH",
    "US-Lin" => "CRO",
    "FR-Pue" => "EBF",
    "IT-Col" => "DBF",
    "US-A10" => "BSV",
    "US-GLE" => "ENF",
    "US-SRC" => "OSH",
    "FI-Si2" => "WET",
    "JP-Mse" => "CRO",
    "RU-Che" => "WET",
    "US-OWC" => "WET",
    "IT-Ro2" => "DBF",
    "US-HRA" => "CRO",
    "FR-LGt" => "WET",
    "JP-SwL" => "WAT",
    "SJ-Adv" => "WET",
    "US-Me2" => "ENF",
    "SE-Deg" => "GRA",
    "FR-Fon" => "DBF",
    "US-MAC" => "WET",
    "US-Whs" => "OSH",
    "JP-MBF" => "DBF",
    "US-Ton" => "WSA",
    "US-Wi9" => "ENF",
    "FI-Lom" => "WET",
    "PH-RiF" => "CRO",
    "US-BZB" => "WET",
    "PA-SPs" => "GRA",
    "US-UMB" => "DBF",
    "GL-ZaH" => "GRA",
    "US-Goo" => "GRA",
    "US-Los" => "WET",
    "US-Beo" => "WET",
    "US-GBT" => "ENF",
)
