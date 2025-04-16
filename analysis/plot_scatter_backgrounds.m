% Scatter plot - metric vs. subjective

D = readtable( '../datasets/AR-DAVID/results_with_parts.csv' );

D.condition_id = cell(height(D),1);
for kk=1:height(D)
    D.condition_id{kk} = strcat( D.scene{kk}, '_', D.distortion{kk}, '_l', num2str(D.level(kk)), '_', num2str(D.luminance(kk)), '_', D.background{kk} );
end

CONFs = { 'mean_cvvdp'};
LABELs = { 'average-lum'};

for ff=1:length(CONFs)
    M = readtable( [ '../metric_results/AR-DAVID-' CONFs{ff} '.csv'] );

J = join( D, M(:,{'condition_id','Q'}), "Keys", "condition_id" );

BKGs = unique( J.background );
LUMs = unique( J.luminance );

figure(1);
html_change_figure_print_size( gcf, 16, 14 );
clf;

min_v = min(min(J.Q), min(J.jod));
max_v = max(max(J.Q), max(J.jod));

% Two plots with all the distortions
cond_cols = { 'distortion', 'scene', 'background', 'luminance' };
colours = lines(6);
symb = 'sop<>^h';

kk = 4;

% connect conditions by lines
K = J;
K.scene_dist = strcat( K.scene, '-', K.distortion, num2str(K.level), K.background );
SDs = unique(K.scene_dist);
for pp=1:length(SDs)
    Kss = K(strcmp(K.scene_dist,SDs{pp}),:);
    Kss = sortrows( Kss, 'jod' );
    plot( Kss.Q, Kss.jod, '-', 'Color', [0.75 0.75 0.75] );
    hold on;
end

plot( [min_v max_v], [min_v max_v], '--k' );
hold on

hh = [];
pp = 1;
for ll=1:length(LUMs)
    for bb=1:length(BKGs)
        ss = strcmp( J.background, BKGs{bb} ) & J.luminance == LUMs(ll);
        label = sprintf( '%s; %g cd/m^2', BKGs{bb}, LUMs(ll) );
        hh(pp) = scatter( J.Q(ss), J.jod(ss), symb(bb), 'MarkerEdgeColor', colours(ll,:), 'MarkerFaceColor', colours(ll,:), 'DisplayName', label );

        pp = pp + 1;
    end
end

legend(hh, 'Location', 'NorthWest');

%                 if vd.do_plot_condition_index
%                     text(J.Q_fixed, J.Q_subj, num2str(Js.condition_index), 'FontSize', 4);
%                 end

xlim( [min_v max_v] );
ylim( [min_v max_v] );

xlabel( 'ColorVideoVDP predictions' );
ylabel( 'Subjective score [JOD]' );
title( LABELs(ff), 'FontWeight','normal');

exportgraphics( gcf, ['scatter_backgrounds_' CONFs{ff} '.pdf'] );

end