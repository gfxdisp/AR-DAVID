% Plot across the backgrounds

D = readtable( '../datasets/AR-DAVID/results_with_parts.csv' );

%D = D(strcmp(D.distortion, 'Dither'),: );

D.bkg_lum = strcat( D.background, '@', num2str( D.luminance ) );


SCNs = unique( D.scene );
DSTs = unique( D.distortion );

BKG_LUMs = unique( D.bkg_lum );

ord = {'flat@ 10', 'leaves@ 10', 'noise@ 10', 'flat@100', 'leaves@100', 'noise@100' };

labels = {'flat@10 cd/m^2', 'leaves@10  cd/m^2', 'noise@10 cd/m^2', 'flat@100 cd/m^2', 'leaves@100 cd/m^2', 'noise@100 cd/m^2' };

figure(1);

COLORs = lines(length(DSTs));
hh = [];
SYBs = 'o<psdh';
LSs = { '--' , '-' };

for cc=1:length(SCNs)

    clf;
    % compute a representative error bar
    ss = strcmp(D.scene, SCNs{cc});
    Dss = D(ss,:);

    eh = sqrt(mean((Dss.jod_high-Dss.jod).^2));
    el = sqrt(mean((Dss.jod-Dss.jod_low).^2));
    % eh = mean(Dss.jod_high-Dss.jod);
    % el = mean(Dss.jod-Dss.jod_low);
    m = mean(Dss.jod );

    errorbar( 0.5, m, el, eh, 'ok' );
    hold on;

    for dd=1:length(DSTs)


        for ll=1:2
            ss = strcmp( D.distortion, DSTs{dd} ) & D.level==(ll+1) & strcmp(D.scene, SCNs{cc});
            Dss = D(ss,:);
            Dss.ord = nan(height(Dss),1);
            for kk=1:height(Dss)
                Dss.ord(kk) = find(strcmp( ord, Dss.bkg_lum{kk} ));
            end
            Dss = sortrows( Dss, 'ord' );

            line_style = [LSs{ll} SYBs(dd)];
            hh(dd) = plot( Dss.ord, Dss.jod, line_style, 'Color', COLORs(dd,:), 'DisplayName', DSTs{dd} );
            hold on
        end

    end
    xticks( 1:6 );
    xticklabels( labels );
    xtickangle( 30 );
    xlim( [0 6.5] );
    ylim( [4.5 12] );
    ylabel( 'Quality [JOD]' );
    title( ['Scene: ', SCNs{cc}] );
    legend( hh, 'Location', 'Best' );

    exportgraphics( gcf, ['ardavid_' SCNs{cc} '.pdf'] );

end