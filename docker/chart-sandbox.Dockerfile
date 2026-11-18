# Provide the verified local wheels with --build-context wheelhouse=<path>.
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

ENV MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/kamidso-matplotlib

COPY chart-sandbox-requirements.txt /tmp/chart-sandbox-requirements.txt
RUN --network=none --mount=type=bind,from=wheelhouse,target=/wheelhouse,readonly \
    python -m pip --isolated --disable-pip-version-check install \
        --no-index --find-links=/wheelhouse --only-binary=:all: --require-hashes \
        --no-cache-dir -r /tmp/chart-sandbox-requirements.txt \
    && python -m pip check

USER 65534:65534
