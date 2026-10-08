import numpy as np
import pickle as pkl
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urljoin,unquote

def get_file_links(session, year, base_url):
    """Return all .txt file URLs listed on a NOAA annual directory page."""
    year_url = urljoin(base_url, f"{year}/")

    response = session.get(year_url, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    links = []

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        url = urljoin(year_url, href)

        if url.lower().endswith(".txt"):
            links.append(url)

    return links


def download_file(session, url, out_dir):
    filename = Path(unquote(url.split("/")[-1])).name
    output_path = out_dir.joinpath(filename)

    # Avoid downloading files that already exist.
    if output_path.exists():
        print(f"already exists: {output_path}")
        return

    with session.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    print(f"saved {output_path}")


def download_stations(start_year, end_year, station_names, out_dir, base_url):
    """
    download all USCRN subhourly text files whose filenames contain
    one of the supplied station-name substrings.
    """
    out_dir = Path(out_dir)
    assert out_dir.exists()

    station_names = [name.lower() for name in station_names]
    with requests.Session() as session:
        for year in range(start_year, end_year + 1):
            try:
                file_links = get_file_links(session, year, base_url)
            except requests.RequestException as exc:
                print(f"could not access {year}: {exc}")
                continue

            for url in file_links:
                filename = Path(unquote(url.split("/")[-1])).name
                filename_lower = filename.lower()

                if any(sn in filename_lower for sn in station_names):
                    download_file(session, url, out_dir)

if __name__=="__main__":
    base_url = "https://www.ncei.noaa.gov/pub/data/uscrn/products/subhourly01/"
    out_dir_annual = Path("/rstor/mdodson/nidis_spie/source/subhourly")
    out_dir_concat = Path("/rstor/mdodson/nidis_spie/source/subhourly_concat")
    start_year = 2006
    end_year = 2026
    get_stations = [
        "AL_Russellville_4_SSE",
        "AL_Fairhope_3_NE",
        "AL_Cullman_3_ENE",
        "AL_Troy_2_W",
        "AL_Talladega_10_NNE",
        "AL_Selma_6_SSE",
        "AL_Scottsboro_2_NE",
        "AL_Thomasville_2_S",
        "AL_Highland_Home_2_S",
        "AL_Gainesville_2_NE",
        "AL_Selma_13_WNW",
        "AL_Valley_Head_1_SSW",
        "AL_Greensboro_2_WNW",
        "AL_Northport_2_S",
        "AL_Courtland_2_WSW",
        "AL_Gadsden_19_N",
        "AL_Brewton_3_NNE",
        "AL_Muscle_Shoals_2_N",
        "AL_Clanton_2_NE",
        ]

    download = False
    concatenate = True

    if download:
        download_stations(
            start_year=start_year,
            end_year=end_year,
            station_names=get_stations,
            out_dir=out_dir_annual,
            base_url=base_url,
            )

    if concatenate:
        sfiles = {}
        for p in out_dir_annual.iterdir():
            _,_,yyyy,sid = p.stem.split("-")
            yyyy = int(yyyy)
            if not sid in get_stations:
                continue
            if not yyyy in range(start_year, end_year+1):
                continue
            if not sid in sfiles.keys():
                sfiles[sid] = []
            sfiles[sid].append((yyyy, p))

        for sid in sfiles.keys():
            sfiles[sid] = list(sorted(sfiles[sid], key=lambda v:v[0]))
            y0 = sfiles[sid][0][0]
            yf = sfiles[sid][-1][0]
            out_path = out_dir_concat.joinpath(f"{sid}-{y0}-{yf}.txt")
            with out_path.open("w") as ofp:
                for _,p in sfiles[sid]:
                    with p.open("r") as ifp:
                        ofp.write(ifp.read())
